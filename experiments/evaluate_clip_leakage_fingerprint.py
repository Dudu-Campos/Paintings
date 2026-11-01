"""A impressao digital do vazamento do CLIP: o zero-shot depende do NOME?

**O que esta ablacao mostra.** Nao e complementaridade -- o nosso banco nao tem
de ser complementar ao CLIP. E **evidencia de que a supervisao do CLIP nomeou os
estilos deste corpus**, que e a ressalva que o §5.18 exige ao vender o canal de
texto (o encoder do estudante e CLIP congelado).

Tres medicoes, e a segunda e a que decide:

1. **zero-shot por classe** -- quais estilos o CLIP acerta so pelo nome. Se o
   vazamento for por nome, os acertos concentram nos movimentos CANONICOS
   (Ukiyo-e, Cubismo, Impressionismo), nao nos rotulos descritivos ou raros.
2. **nulo por EMBARALHAMENTO do nome.** O §5.18b registra o erro de usar o acaso
   teorico como nulo: a primeira guarda de alinhamento abortou os dois corpora
   com o alinhamento CORRETO porque comparava contra colisao teorica. O nulo
   certo aqui e permutar a atribuicao nome->classe mantendo os MESMOS 125 textos
   e as MESMAS imagens: se o zero-shot cair para o acaso, o que trabalha e a
   identidade da palavra, nao a plausibilidade generica do template.
3. **template sem estilo** -- `"a painting"` igual para todas as classes, que
   por construcao nao carrega informacao de classe. Piso de sanidade: tem de dar
   o acaso exato.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/pipeline"))
from protocol import SEED  # noqa: E402

CLIP_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPLATE = "a painting in the style of {}"


def log(m: str) -> None:
    print(m, flush=True)


def readable(style: str) -> str:
    text = unicodedata.normalize("NFKD", style)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s*\(([^)]*)\)", r" \1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    ds = args.dataset
    args.output = args.output or ROOT / f"results/{ds}_clip_leakage.json"

    frame = pd.read_csv(ROOT / f"results/{ds}_student_captions.csv",
                        dtype={"stem": str, "style": str})
    feats = np.load(ROOT / f"outputs/{ds}_clip_vitb32.npy")
    if len(feats) != len(frame):
        raise SystemExit(f"desalinhamento: {len(feats):,} x {len(frame):,}")
    labels = frame["style"].to_numpy(str)
    classes = np.unique(labels)
    emb = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    chance = 1.0 / len(classes)
    log(f"{ds}: {len(frame):,} imagens | {len(classes)} classes | acaso {100*chance:.2f}%")

    from transformers import CLIPModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
    model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()

    def encode(texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            batch = tok(texts, padding=True, return_tensors="pt").to(DEVICE)
            out = model.get_text_features(**batch)
        # Nesta versao do transformers `get_text_features` devolve um
        # BaseModelOutputWithPooling, e o `pooler_output` dele JA E o embedding
        # projetado -- conferido por `allclose` contra
        # `text_projection(text_model(...).pooler_output)`. O nome engana: nao e
        # o pooled pre-projecao. Sem esta guarda o script quebra; com ela usando
        # o vetor errado, as dimensoes ainda casariam (512 dos dois lados no
        # ViT-B/32) e o zero-shot sairia baixo EM SILENCIO.
        if not isinstance(out, torch.Tensor):
            out = out.pooler_output
        vec = out.cpu().numpy().astype(np.float32)
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)

    text = encode([TEMPLATE.format(readable(c)) for c in classes])
    scores = emb @ text.T
    predicted = classes[scores.argmax(1)]
    real = balanced_accuracy_score(labels, predicted)
    log(f"\n  zero-shot REAL          BAcc {100*real:5.2f}  ({real/chance:.1f}x o acaso)")

    # --- nulo 1: permutar a atribuicao nome -> classe ---------------------
    # Os MESMOS 125 textos e as MESMAS imagens; so o pareamento e destruido.
    # Este e o nulo que o §5.18b prescreve, em vez do acaso teorico.
    rng = np.random.default_rng(SEED)
    nulos = []
    for _ in range(args.permutations):
        troca = rng.permutation(len(classes))
        nulos.append(balanced_accuracy_score(labels, classes[troca][scores.argmax(1)]))
    nulos = np.array(nulos)
    log(f"  nulo EMBARALHADO        BAcc {100*nulos.mean():5.2f} +- {100*nulos.std():.2f} "
        f"({args.permutations} permutacoes)")

    # --- nulo 2: template sem estilo, identico para toda classe -----------
    flat = encode(["a painting"] * len(classes))
    plano = balanced_accuracy_score(labels, classes[(emb @ flat.T).argmax(1)])
    log(f"  nulo SEM ESTILO         BAcc {100*plano:5.2f}  (piso de sanidade)")
    log(f"  razao real / embaralhado: {real/max(nulos.mean(), 1e-9):.1f}x")

    # --- por classe -------------------------------------------------------
    recalls = recall_score(labels, predicted, labels=classes, average=None, zero_division=0)
    counts = pd.Series(labels).value_counts()
    por_classe = pd.DataFrame({
        "style": classes, "n": [int(counts[c]) for c in classes],
        "prompt": [readable(c) for c in classes], "zero_shot_recall": 100 * recalls,
    }).sort_values("zero_shot_recall", ascending=False)
    log(f"\n  classes com recall zero-shot > 0: "
        f"{int((recalls > 0).sum())} de {len(classes)}")
    log("\n  --- os 15 que o CLIP acerta SO PELO NOME ---")
    for _, r in por_classe.head(15).iterrows():
        log(f"    {r['prompt'][:34]:34s} n={r['n']:5d}  recall {r['zero_shot_recall']:5.1f}")

    por_classe.to_csv(ROOT / f"results/{ds}_clip_zeroshot_by_class.csv", index=False)
    args.output.write_text(json.dumps({
        "dataset": ds, "n_classes": int(len(classes)), "chance": chance,
        "zero_shot_bacc": float(real),
        "null_shuffled_mean": float(nulos.mean()), "null_shuffled_std": float(nulos.std()),
        "null_flat_template_bacc": float(plano),
        "ratio_real_over_shuffled": float(real / max(nulos.mean(), 1e-9)),
        "classes_with_nonzero_recall": int((recalls > 0).sum()),
    }, indent=2))
    log(f"\nsalvo: {args.output}")


if __name__ == "__main__":
    main()
