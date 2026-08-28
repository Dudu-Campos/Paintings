"""Etapa 05 — avaliação.

Dois blocos:

1. Fidelidade de destilação no split de TESTE do COCO: as legendas do estudante
   vs. as do professor (BLEU-4, ROUGE-L, CIDEr/METEOR se disponíveis) e
   concordância de sentimento (classificador de afeto sobre ambas).

2. Comparação com o CATR sobre o WikiArt: rodar o estudante nas imagens do
   WikiArt e confrontar com o baseline CATR/COCO do projeto — tanto qualitativo
   (exemplos lado a lado) quanto downstream, se plugado no ramo TF-IDF/fusão.
   Aqui também cabe reusar o leak_probe para confirmar que o estudante NÃO
   emite artista/estilo (controle limpo por construção).

Início/fim na inferência do estudante:
  - parada por <eos> E por max_len (fallback);
  - no_repeat_ngram_size=3 e repetition_penalty~1.2 (decoders pequenos entram
    em loop sem isso); beam search (3–5) acerta melhor o EOS.

Entrada:
    artifacts/student_ckpt/          (etapa 04)
    artifacts/tokenizer/             (etapa 03)
    artifacts/image_index.csv        (split=test do COCO)
    data/wikiart/ + legendas CATR    (baseline do projeto)

Saída:
    artifacts/eval_report.csv        métricas por conjunto
    artifacts/eval_report.md         resumo legível + exemplos lado a lado

Uso:
    python 05_evaluate.py --beam 5 --max-len 64
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    BOS,
    D_FF,
    D_MODEL,
    DROPOUT,
    EOS,
    MAX_CAPTION_TOKENS,
    N_HEADS,
    N_LAYERS,
    OUT_DIR,
    PAD,
    STUDENT_CKPT_DIR,
    TEACHER_CAPTIONS_CSV,
    TOKENIZER_DIR,
    log,
)

# ------------------------------------------------------------------
# Reuso do modelo/CLIP da etapa 04 (o prefixo numérico impede import direto)
# ------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "student_stage", str(Path(__file__).resolve().parent / "04_student.py"))
student_stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(student_stage)
VisualPrefixDecoder = student_stage.VisualPrefixDecoder
build_clip_cache = student_stage.build_clip_cache
decode_clean = student_stage.decode_clean
CLIP_DIM = student_stage.CLIP_DIM
DEVICE = student_stage.DEVICE

WORD_RE = re.compile(r"[a-z0-9]+")

# 27 estilos do WikiArt -> tokens de rótulo (para scrub anti-vazamento)
WIKIART_STYLES = [
    "Abstract_Expressionism", "Action_painting", "Analytical_Cubism",
    "Art_Nouveau_Modern", "Baroque", "Color_Field_Painting",
    "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism",
    "Fauvism", "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance",
    "Minimalism", "Naive_Art_Primitivism", "New_Realism", "Northern_Renaissance",
    "Pointillism", "Pop_Art", "Post_Impressionism", "Realism", "Rococo",
    "Romanticism", "Symbolism", "Synthetic_Cubism", "Ukiyo_e",
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(WORD_RE.findall(text.lower()))


def style_label_tokens() -> set[str]:
    """Todos os tokens lexicais que compõem os nomes dos 27 estilos +
    boilerplate de arte — removidos das legendas antes do TF-IDF, para que o
    classificador não veja o rótulo escrito na legenda."""
    toks: set[str] = set()
    for s in WIKIART_STYLES:
        for t in normalize(s.replace("_", " ")).split():
            toks.add(t)
    # sinônimos/variações comuns que colam no rótulo
    toks |= {"renaissance", "cubist", "impressionist", "baroque", "rococo",
             "expressionist", "romantic", "realist", "abstract", "modern",
             "art", "painting", "paintings", "artwork", "style", "movement",
             "century", "painted", "painter"}
    return toks


# ------------------------------------------------------------------
# Carregar estudante treinado
# ------------------------------------------------------------------
def load_student(ckpt_path: Path, vocab_size: int, pad_id: int):
    model = VisualPrefixDecoder(
        vocab_size, D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT,
        student_stage.PREFIX_LEN if hasattr(student_stage, "PREFIX_LEN") else 10,
        CLIP_DIM, 512, pad_id).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    log(f"estudante carregado: {ckpt_path} (ep {state.get('epoch')}, "
        f"ROUGE-L {state.get('rougeL')})")
    return model


def gen_captions(model, feats, tok, bos_id, eos_id, max_len, batch=256):
    """Gera legendas em lote com a decodificação do próprio modelo."""
    out = []
    for i in range(0, len(feats), batch):
        chunk = torch.tensor(feats[i:i + batch], dtype=torch.float32, device=DEVICE)
        seqs = model.generate(chunk, bos_id, eos_id, max_len)
        out.extend(decode_clean(tok, s, bos_id, eos_id) for s in seqs)
        log(f"  gerado {min(i + batch, len(feats))}/{len(feats)}")
    return out


# ------------------------------------------------------------------
# Bloco 1: fidelidade de destilação (BLEU-4 + ROUGE-L vs professor)
# ------------------------------------------------------------------
def bleu4(ref: str, hyp: str) -> float:
    """BLEU-4 com suavização (sentence-level, brevity penalty)."""
    r, h = ref.split(), hyp.split()
    if not h:
        return 0.0
    weights = 0.0
    for n in range(1, 5):
        ref_ng = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        hyp_ng = collections.Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        overlap = sum((hyp_ng & ref_ng).values())
        total = max(1, sum(hyp_ng.values()))
        # suavização +1 (evita zero em n-gramas altos)
        p = (overlap + 1) / (total + 1)
        weights += math.log(p)
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / max(1, len(h)))
    return bp * math.exp(weights / 4)


def fidelity_coco(model, tok, bos_id, eos_id, pad_id, max_len, limit=0):
    caps = pd.read_csv(TEACHER_CAPTIONS_CSV)
    test = caps[caps["split"] == "test"].reset_index(drop=True)
    if limit:
        test = test.head(limit)
    log(f"Bloco 1 — fidelidade COCO test: {len(test)} imagens")
    feats = build_clip_cache(list(test["path"]), OUT_DIR / "clip_feats_test.npy")
    hyps = gen_captions(model, feats, tok, bos_id, eos_id, max_len)
    refs = [str(c).replace(EOS, "").strip() for c in test["caption"]]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouges = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    bleus = [bleu4(r, h) for r, h in zip(refs, hyps)]
    examples = [{"path": test.iloc[i]["file_name"], "professor": refs[i],
                 "estudante": hyps[i]} for i in range(min(8, len(hyps)))]
    return {
        "n": len(hyps),
        "bleu4": float(np.mean(bleus)),
        "rougeL": float(np.mean(rouges)),
    }, examples


# ------------------------------------------------------------------
# Bloco 2a: auditoria de vazamento (mesmo protocolo do catr_coco_leakage)
# ------------------------------------------------------------------
def leakage_audit(df: pd.DataFrame) -> dict:
    """df tem colunas: style (nome do dir), artist, caption."""
    n = len(df)
    true_style_hits = 0
    any_style_hits = 0
    surname_hits = 0
    style_tok = {s: set(normalize(s.replace("_", " ")).split()) for s in WIKIART_STYLES}
    all_style_tok = set().union(*style_tok.values())
    for _, row in df.iterrows():
        cap = normalize(row["caption"])
        cap_set = set(cap.split())
        # menção ao estilo verdadeiro (qualquer token do nome do estilo)
        if style_tok.get(row["style"], set()) & cap_set:
            true_style_hits += 1
        # menção a QUALQUER estilo
        if all_style_tok & cap_set:
            any_style_hits += 1
        # menção ao sobrenome do artista
        artist = normalize(row.get("artist", ""))
        surname = artist.split()[-1] if artist and artist != "nan" else ""
        if surname and len(surname) > 2 and surname in cap_set:
            surname_hits += 1
    return {
        "n": n,
        "true_style_mention_pct": 100 * true_style_hits / max(1, n),
        "any_style_mention_pct": 100 * any_style_hits / max(1, n),
        "surname_mention_pct": 100 * surname_hits / max(1, n),
    }


# ------------------------------------------------------------------
# Bloco 2b: classificação downstream TF-IDF (protocolo do projeto)
# ------------------------------------------------------------------
def scrub(caption: str, stop: set[str]) -> str:
    """Remove tokens de rótulo/boilerplate para o classificador não ler o
    estilo escrito na legenda (mesma defesa do tfidf_caption_style)."""
    toks = [t for t in normalize(caption).split() if t not in stop]
    return " ".join(toks)


def tfidf_classify(df: pd.DataFrame, seed: int = 42) -> dict:
    """TF-IDF binário ngram(1,2) min_df=10 max_df=0.90 max_features=50000 +
    LinearSVC L1 C=0.1 class_weight=balanced, 5-fold estratificado, BAcc OOF.
    Vocabulário ajustado DENTRO de cada fold de treino (sem vazamento)."""
    stop = style_label_tokens()
    texts = np.array([scrub(c, stop) for c in df["caption"]])
    labels = df["style"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_pred = np.empty(len(labels), dtype=object)
    fold_baccs = []
    for tr, te in skf.split(texts, labels):
        vec = TfidfVectorizer(binary=True, ngram_range=(1, 2), min_df=10,
                              max_df=0.90, max_features=50000)
        xtr = vec.fit_transform(texts[tr])
        xte = vec.transform(texts[te])
        clf = LinearSVC(penalty="l1", C=0.1, class_weight="balanced", dual=False)
        clf.fit(xtr, labels[tr])
        pred = clf.predict(xte)
        oof_pred[te] = pred
        fold_baccs.append(balanced_accuracy_score(labels[te], pred))
    return {
        "n_images": len(labels),
        "n_styles": len(np.unique(labels)),
        "bacc_oof": float(balanced_accuracy_score(labels, oof_pred)),
        "bacc_fold_mean": float(np.mean(fold_baccs)),
        "bacc_fold_std": float(np.std(fold_baccs)),
    }


# ------------------------------------------------------------------
# Geração no WikiArt (a partir de um índice já preparado)
# ------------------------------------------------------------------
def wikiart_index(wikiart_root: Path, subset: str, limit: int, seed: int):
    classes = pd.read_csv(wikiart_root / "classes.csv")
    classes = classes[classes["filename"].apply(
        lambda f: (wikiart_root / f).is_file())]
    if subset:
        classes = classes[classes["subset"].astype(str) == subset]
    classes = classes.copy()
    classes["style"] = classes["filename"].apply(lambda f: str(Path(f).parts[0]))
    if limit:
        # amostra estratificada por estilo (por-grupo, sem groupby.apply que
        # consome a coluna de agrupamento no pandas 3)
        per = max(1, limit // classes["style"].nunique())
        parts = [g.sample(min(len(g), per), random_state=seed)
                 for _, g in classes.groupby("style")]
        classes = pd.concat(parts, ignore_index=True)
    classes["path"] = classes["filename"].apply(lambda f: str(wikiart_root / f))
    return classes[["filename", "style", "artist", "path"]].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=STUDENT_CKPT_DIR / "best.pt")
    parser.add_argument("--wikiart-root", type=Path, required=True)
    parser.add_argument("--subset", default="test")
    parser.add_argument("--max-len", type=int, default=MAX_CAPTION_TOKENS)
    parser.add_argument("--limit-wikiart", type=int, default=0,
                        help="0 = todas as imagens do subset")
    parser.add_argument("--limit-coco", type=int, default=0)
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--only-generate", action="store_true",
                        help="só gera legendas WikiArt e salva o CSV")
    parser.add_argument("--captions-csv", type=Path,
                        help="reusa legendas WikiArt já geradas (pula geração)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    bos_id, eos_id, pad_id = (tok.token_to_id(BOS), tok.token_to_id(EOS),
                              tok.token_to_id(PAD))
    vocab_size = tok.get_vocab_size()
    model = load_student(args.ckpt, vocab_size, pad_id)

    report = {}
    coco_examples = []

    # --- Bloco 1: fidelidade COCO test ---
    if not args.skip_coco:
        fid, coco_examples = fidelity_coco(
            model, tok, bos_id, eos_id, pad_id, args.max_len, args.limit_coco)
        report["coco_fidelity"] = fid
        log(f"COCO fidelidade: BLEU-4={fid['bleu4']:.4f} "
            f"ROUGE-L={fid['rougeL']:.4f} (n={fid['n']})")

    # --- Bloco 2: WikiArt ---
    if args.captions_csv and args.captions_csv.is_file():
        wa = pd.read_csv(args.captions_csv)
        log(f"Reusando legendas WikiArt: {len(wa)} imagens")
    else:
        idx = wikiart_index(args.wikiart_root, args.subset,
                            args.limit_wikiart, args.seed)
        log(f"Bloco 2 — WikiArt {args.subset}: {len(idx)} imagens")
        feats = build_clip_cache(list(idx["path"]),
                                 OUT_DIR / f"clip_feats_wikiart_{args.subset}.npy")
        caps = gen_captions(model, feats, tok, bos_id, eos_id, args.max_len)
        wa = idx.copy()
        wa["caption"] = caps
        wa_path = OUT_DIR / "student_wikiart_captions.csv"
        wa.to_csv(wa_path, index=False)
        log(f"Escrito {wa_path}")

    if args.only_generate:
        return

    # auditoria de vazamento
    leak = leakage_audit(wa)
    report["leakage"] = leak
    log(f"Vazamento estudante: estilo_verdadeiro={leak['true_style_mention_pct']:.3f}% "
        f"qualquer_estilo={leak['any_style_mention_pct']:.3f}% "
        f"sobrenome={leak['surname_mention_pct']:.3f}%")

    # classificação downstream
    clf = tfidf_classify(wa, args.seed)
    report["tfidf_downstream"] = clf
    log(f"BAcc downstream estudante: {clf['bacc_oof']:.4f} "
        f"(fold {clf['bacc_fold_mean']:.4f}±{clf['bacc_fold_std']:.4f})")

    # exemplos WikiArt lado a lado (estudante), 2 por alguns estilos
    wa_examples = (wa.groupby("style").head(1).head(10)
                   [["style", "artist", "caption"]].to_dict("records"))

    # --- referências CATR do projeto (canônicas) ---
    report["catr_reference"] = {
        "bacc_oof": 0.18138, "n_images": 12210,
        "true_style_mention_pct": 0.0, "surname_mention_pct": 0.0,
        "note": "results/catr_coco_standalone_results.csv + catr_coco_leakage_summary.csv",
    }

    # --- salvar relatório ---
    rows = []
    if "coco_fidelity" in report:
        rows.append({"bloco": "fidelidade_coco", "metric": "bleu4",
                     "value": report["coco_fidelity"]["bleu4"]})
        rows.append({"bloco": "fidelidade_coco", "metric": "rougeL",
                     "value": report["coco_fidelity"]["rougeL"]})
    rows.append({"bloco": "vazamento_estudante", "metric": "true_style_pct",
                 "value": leak["true_style_mention_pct"]})
    rows.append({"bloco": "vazamento_estudante", "metric": "surname_pct",
                 "value": leak["surname_mention_pct"]})
    rows.append({"bloco": "downstream", "metric": "bacc_estudante",
                 "value": clf["bacc_oof"]})
    rows.append({"bloco": "downstream", "metric": "bacc_catr_ref",
                 "value": 0.18138})
    pd.DataFrame(rows).to_csv(OUT_DIR / "eval_report.csv", index=False)
    (OUT_DIR / "eval_report.json").write_text(json.dumps({
        "report": report, "coco_examples": coco_examples,
        "wikiart_examples": wa_examples}, indent=2, ensure_ascii=False))
    log(f"Relatório salvo em {OUT_DIR / 'eval_report.csv'} + .json")


if __name__ == "__main__":
    main()
