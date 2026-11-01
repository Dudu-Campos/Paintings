"""Os dois bracos de CLIP da ablacao de captioner (CLIP x Qwen x estudante).

O §5.18 registra que a diferenca entre o probe do embedding do CLIP e os ~16pp
do canal de texto **nao e identificavel**: ela e igualmente compativel com "a
linguagem descarta a supervisao de estilo" e com "o nosso decoder e ruim". Estes
dois bracos sao o lado CLIP da separacao, e medem coisas diferentes:

* **zero-shot** -- cosseno entre a imagem e os 125 textos
  `"a painting in the style of <estilo>"`. NAO usa rotulo do corpus. So funciona
  se a imagem estiver alinhada a **palavra** do estilo, o que e evidencia direta
  de que a supervisao da web nomeou os estilos. Probe linear nao prova isso:
  ele so mostra que a informacao e linearmente decodificavel.
* **probe linear** -- LinearSVC sobre as 512 dims, MESMO protocolo do canal de
  texto (5 folds, OOF, class_weight balanceado, SEED 42). E o teto do que o
  embedding carrega, com a mesma supervisao que o canal de texto recebe.

Os dois usam o cache `outputs/<ds>_clip_vitb32.npy`, produzido pelo
`build_clip_cache` do estudante com `openai/clip-vit-base-patch32` -- os MESMOS
pesos do encoder do estudante. Trocar de pesos aqui mediria outro encoder e a
comparacao com o estudante deixaria de isolar o decoder.
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/pipeline"))
from protocol import N_FOLDS, SEED  # noqa: E402

CLIP_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Um template so, declarado. A literatura de CLIP usa ensemble de ~80 prompts, o
# que sobe o zero-shot alguns pontos -- mas a pergunta aqui NAO e "qual o melhor
# zero-shot possivel", e sim "a supervisao da web nomeou os estilos?". Um
# template fixo responde isso e nao tem grau de liberdade para ser afinado.
TEMPLATE = "a painting in the style of {}"


def log(message: str) -> None:
    print(message, flush=True)


def readable(style: str) -> str:
    """`Color_Field_Painting` -> `color field painting`.

    Os rotulos do Multitask tem underscore, parenteses e acento (§8 registra os
    tres eixos de divergencia). O CLIP viu texto natural, entao o nome tem de
    chegar como frase.
    """
    text = unicodedata.normalize("NFKD", style)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s*\(([^)]*)\)", r" \1", text)      # `Naive Art (Primitivism)`
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="multitask")
    parser.add_argument("--captions", type=Path, default=None,
                        help="CSV do estudante: fonte canonica de stem+style+ordem")
    parser.add_argument("--clip-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--clip-model", default=CLIP_NAME,
                        help="outro CLIP exige cache de imagem do MESMO modelo")
    parser.add_argument("--skip-probe", action="store_true",
                        help="so o zero-shot; o probe e o caro")
    args = parser.parse_args()
    ds = args.dataset
    args.captions = args.captions or ROOT / f"results/{ds}_student_captions.csv"
    args.clip_cache = args.clip_cache or ROOT / f"outputs/{ds}_clip_vitb32.npy"
    args.output = args.output or ROOT / f"results/{ds}_clip_text_arms.json"

    frame = pd.read_csv(args.captions, dtype={"stem": str, "style": str})
    feats = np.load(args.clip_cache)
    # O cache e o CSV saem do MESMO `meta` na mesma execucao do
    # evaluate_student_captions_multitask.py, entao a linha i corresponde. Se um
    # dos dois for regerado sozinho isso deixa de valer EM SILENCIO -- o §5.18b
    # mede que um bloco de texto embaralhado custa -2,92pp sem reclamar.
    if len(feats) != len(frame):
        raise SystemExit(f"desalinhamento: {len(feats):,} features contra "
                         f"{len(frame):,} legendas. Regerar os dois juntos.")
    labels = frame["style"].to_numpy(str)
    classes = np.unique(labels)
    log(f"{ds}: {len(frame):,} imagens | {len(classes)} classes | CLIP {feats.shape[1]}d")

    emb = feats / np.linalg.norm(feats, axis=1, keepdims=True)

    # --- braco 1: zero-shot, SEM rotulo do corpus -------------------------
    from transformers import CLIPModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(DEVICE).eval()
    prompts = [TEMPLATE.format(readable(c)) for c in classes]
    log(f"exemplos de prompt: {prompts[0]!r} ... {prompts[-1]!r}")
    with torch.inference_mode():
        batch = tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
        text = model.get_text_features(**batch)
        if not isinstance(text, torch.Tensor):
            text = text.pooler_output
    text = text.cpu().numpy().astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)

    predicted = classes[(emb @ text.T).argmax(1)]
    zero_shot = balanced_accuracy_score(labels, predicted)

    # O zero-shot NAO treina, mas isso nao o deixa sem barra: folds diferentes
    # tem imagens diferentes, e a variacao de AMOSTRAGEM e da ordem de 0,5pp
    # com ~20 mil imagens por fold (B/32 vai de 11,15 a 12,62). Reportar celula
    # vazia sugeria "nao existe" quando o certo era "nao medi". A particao e a
    # MESMA do probe, entao as duas barras sao computadas sobre os mesmos cortes.
    zs_folds = [balanced_accuracy_score(labels[te], predicted[te])
                for _, te in StratifiedKFold(N_FOLDS, shuffle=True,
                                             random_state=SEED).split(emb, labels)]
    zs_acc = accuracy_score(labels, predicted)
    zs_f1 = f1_score(labels, predicted, average="macro", zero_division=0)
    # a predicao por imagem e o que faltava para o macro-F1: a tabela por classe
    # so carrega recall, e precisao exige saber o que foi predito onde.
    np.save(args.output.with_name(args.output.stem + "_zeroshot_pred.npy"),
            predicted.astype(str))
    chance = 1.0 / len(classes)
    log(f"  zero-shot        BAcc {100*zero_shot:5.2f}  ({zero_shot/chance:.1f}x o acaso)")
    log(f"  zero-shot folds  BAcc {100*np.mean(zs_folds):5.2f} +- "
        f"{100*np.std(zs_folds, ddof=1):.2f}  (variacao de AMOSTRAGEM, nao de treino)")

    # --- braco 2: probe linear, MESMO protocolo do canal de texto ---------
    probe_bacc = probe_acc = probe_f1 = float("nan")
    per_fold = [float("nan")]
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
                  .split(emb, labels)) if not args.skip_probe else []
    oof = np.empty(len(labels), dtype=object)
    if splits:
        per_fold = []
    for k, (train, test) in enumerate(splits, 1):
        probe = LinearSVC(class_weight="balanced", random_state=SEED)
        probe.fit(emb[train], labels[train])
        oof[test] = probe.predict(emb[test])
        per_fold.append(balanced_accuracy_score(labels[test], oof[test]))
        log(f"  probe fold {k}     BAcc {100*per_fold[-1]:5.2f}")
    if splits:
        # A predicao OOF precisa ir para disco pelo mesmo motivo da do zero-shot:
        # sem ela as colunas Acc/macro-F1 ficam sem barra por fold e a unica
        # saida e reexecutar o probe inteiro (~20 min por modelo).
        np.save(args.output.with_name(args.output.stem + "_probe_oof.npy"),
                oof.astype(str))
        probe_bacc = balanced_accuracy_score(labels, oof.astype(str))
        probe_acc = accuracy_score(labels, oof.astype(str))
        probe_f1 = f1_score(labels, oof.astype(str), average="macro", zero_division=0)
        log(f"  probe linear     BAcc {100*probe_bacc:5.2f} "
            f"(folds {100*np.mean(per_fold):.2f} +- "
            f"{100*np.std(per_fold, ddof=1):.2f})")

    report = {
        "dataset": ds, "n_images": int(len(frame)), "n_styles": int(len(classes)),
        "template": TEMPLATE, "clip_model": args.clip_model,
        "zero_shot_bacc": float(zero_shot),
        "zero_shot_x_chance": float(zero_shot / chance),
        "zero_shot_acc": float(zs_acc), "zero_shot_macro_f1": float(zs_f1),
        "zero_shot_fold_mean": float(np.mean(zs_folds)),
        "zero_shot_fold_std": float(np.std(zs_folds, ddof=1)),
        "probe_acc": float(probe_acc), "probe_macro_f1": float(probe_f1),
        "probe_bacc_oof": float(probe_bacc),
        "probe_bacc_fold_mean": float(np.mean(per_fold)),
        "probe_bacc_fold_std": float(np.std(per_fold, ddof=1)),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"salvo: {args.output}")


if __name__ == "__main__":
    main()
