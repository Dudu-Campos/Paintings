"""theta-robustez da curva de Hilbert: theta* é um pico isolado?

A pergunta NÃO é se a transformação é robusta a qualquer configuração -- ela
vem da literatura e não é nossa para defender. A pergunta é se o ponto de
operação escolhido a priori (`N_SIZE=128`, `N_BANDS=8`) está num ponto
não-suave da superfície de resposta, o que tornaria todos os resultados
dependentes de uma configuração afortunada.

Previsão teórica que torna o Hilbert o primeiro caso a testar: as estatísticas
de banda são médias sobre N^2/B pixels, e a curva preserva localidade, então
cada banda cobre **a mesma região da imagem** em qualquer resolução. Se a teoria
vale, `n_size` quase não deveria mover nada.

Duas métricas, porque medem coisas diferentes:

  intrínseca  CKA linear entre sigma(theta) e sigma(theta*), SEM classificador.
              É a definição direta de "implementações admissíveis medem a mesma
              coisa", e não herda a sensibilidade a classificador que o §5.12
              acabou de documentar (trocar MLP estreito por largo inverteu o
              sinal de um efeito de 0,79pp).
  extrínseca  BAcc isolada do bloco, LDA e MLP, mais o custo de extração.

**Ressalva de escala que a figura tem de carregar:** a mediana do lado maior no
Multitask é 348 px, então `n_size` acima disso é interpolação. A estabilidade
observada ali não é propriedade da transformação -- é upsampling não criando
informação. A fronteira sai no CSV como `above_native`.

O relatório inclui o **rank de theta\\* dentro da grade**. Se theta* sair como o
melhor, isso é evidência de tuning e enfraquece a seção em vez de apoiá-la; o
resultado que sustenta a alegação é theta* **típico**, não ótimo.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from evaluate_neighbour_attention import key  # noqa: E402

PYTHON = ROOT / ".ABSvenv/bin/python3"
STAR = (128, 8)          # theta* de produção
NATIVE_MEDIAN = 348      # mediana do lado maior da coorte, medida com PIL


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """CKA linear. Aceita dimensionalidades diferentes, que é o motivo de ser a
    métrica intrínseca aqui: `n_bands` muda o número de colunas."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    xy = np.linalg.norm(x.T @ y, "fro") ** 2
    xx = np.linalg.norm(x.T @ x, "fro")
    yy = np.linalg.norm(y.T @ y, "fro")
    return float(xy / (xx * yy)) if xx > 0 and yy > 0 else 0.0


def extract(image_dir: Path, meta: Path, out: Path, n_size: int, n_bands: int,
            workers: int) -> float:
    marker = out / "hilbert_features.npy"
    if marker.exists():
        return float("nan")
    started = time.perf_counter()
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "src/extractors/extract_hilbert_features.py"),
         "--image_dir", str(image_dir), "--meta", str(meta), "--output", str(out),
         "--workers", str(workers), "--n-size", str(n_size), "--n-bands", str(n_bands)],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"extração falhou ({n_size},{n_bands}):\n{result.stderr[-2000:]}")
    return time.perf_counter() - started


def load(out: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(out / "hilbert_features.npy")
    stems = np.load(out / "hilbert_stems.npy", allow_pickle=True).astype(str)
    return values, stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "data/multitask_painting100k/images")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/multitask_meta.csv")
    parser.add_argument("--classes-file", type=Path,
                        default=ROOT / "data/multitask_25class_benchmark.txt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--bands", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/theta_hilbert")
    parser.add_argument("--output", type=Path, default=ROOT / "results/theta_hilbert.csv")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    wanted = {key(l) for l in args.classes_file.read_text().splitlines() if l.strip()}
    # `meta.style` devolve o acessor Styler do pandas, não a coluna (§8).
    meta = meta[[key(str(v)) in wanted for v in meta["style"]]]
    if meta["style"].nunique() != len(wanted):
        raise ValueError(f"casaram {meta['style'].nunique()} de {len(wanted)} classes")
    log(f"coorte do benchmark: {len(meta):,} imagens, {meta['style'].nunique()} classes")

    grid = [(s, b) for s in args.sizes for b in args.bands]
    features, costs = {}, {}
    for n_size, n_bands in grid:
        out = args.work_dir / f"n{n_size}_b{n_bands}"
        seconds = extract(args.image_dir, args.meta, out, n_size, n_bands, args.workers)
        costs[(n_size, n_bands)] = seconds
        features[(n_size, n_bands)] = load(out)
        log(f"  ({n_size:>3}, {n_bands:>2})  {features[(n_size, n_bands)][0].shape}  "
            f"{seconds:.0f}s" if seconds == seconds else
            f"  ({n_size:>3}, {n_bands:>2})  em cache")

    # Alinhar tudo pela coorte do benchmark, na mesma ordem de linhas.
    order = meta.stem.astype(str).to_numpy()
    aligned = {}
    for theta, (values, stems) in features.items():
        index = pd.Series(np.arange(len(stems)), index=stems)
        take = index.reindex(order).dropna()
        aligned[theta] = (values[take.to_numpy(dtype=int)], take.index.to_numpy())
    common = set(aligned[STAR][1])
    for theta in aligned:
        common &= set(aligned[theta][1])
    common = np.array(sorted(common))
    log(f"linhas comuns a todas as configurações: {len(common):,}")
    matrices = {}
    for theta, (values, stems) in aligned.items():
        index = pd.Series(np.arange(len(stems)), index=stems)
        matrices[theta] = values[index.reindex(common).to_numpy(dtype=int)]
    labels = LabelEncoder().fit_transform(
        meta.set_index("stem").loc[common, "style"].to_numpy(dtype=str))
    n_classes = len(np.unique(labels))

    reference = matrices[STAR]
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42)
                  .split(np.zeros(len(labels)), labels))[:args.folds]

    rows = []
    for theta, values in matrices.items():
        cka = linear_cka(StandardScaler().fit_transform(values),
                         StandardScaler().fit_transform(reference))
        scores = []
        for train, test in splits:
            scaler = StandardScaler().fit(values[train])
            model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
            model.fit(scaler.transform(values[train]), labels[train])
            scores.append(100 * balanced_accuracy_score(
                labels[test], model.predict(scaler.transform(values[test]))))
        rows.append({"n_size": theta[0], "n_bands": theta[1], "dimensions": values.shape[1],
                     "is_star": theta == STAR, "above_native": theta[0] > NATIVE_MEDIAN,
                     "cka_vs_star": cka, "lda_bacc": float(np.mean(scores)),
                     "lda_std": float(np.std(scores)), "extract_seconds": costs[theta]})
        log(f"  ({theta[0]:>3}, {theta[1]:>2})  CKA {cka:.4f}  LDA {np.mean(scores):5.2f}")
        pd.DataFrame(rows).to_csv(args.output, index=False)

    frame = pd.DataFrame(rows).sort_values("lda_bacc", ascending=False).reset_index(drop=True)
    star_rank = int(frame.index[frame.is_star][0]) + 1
    log(f"\ntheta* = {STAR} é o {star_rank}º de {len(frame)} em BAcc "
        f"(se fosse o 1º, seria evidência de tuning, não de robustez)")
    log("\n" + pd.DataFrame(rows).pivot_table(
        index="n_size", columns="n_bands", values="lda_bacc").round(2).to_string())
    log("\nCKA contra theta*:")
    log(pd.DataFrame(rows).pivot_table(
        index="n_size", columns="n_bands", values="cka_vs_star").round(3).to_string())


if __name__ == "__main__":
    main()
