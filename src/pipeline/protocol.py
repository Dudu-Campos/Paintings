"""O protocolo de avaliação: constantes, splits e as duas sondas.

**Por que este arquivo existe.** As constantes que definem o protocolo do
projeto inteiro estavam espalhadas por scripts de experimento: `SEED = 42` em
`experiments/fusion_curve_slic.py` (um experimento de curva de fusão do SLIC),
`N_FOLDS = 5` em `experiments/final_tfidf_visual_mlp.py` (um experimento de
TF-IDF, importado por **62** outros scripts), e `evaluate_mlp`/`evaluate_lda` em
`experiments/evaluate_full_factorial_ablation.py` (51 importadores). O §2 do
CLAUDE.md declara essas convenções como se fossem do projeto; agora elas moram
num lugar que corresponde a essa declaração.

**As duas sondas medem coisas diferentes e a diferença é diagnóstico.** O §8
registra: LDA e MLP concordam em <=0,35pp em vetores densos e divergem ~9pp em
vetores multi-bloco -- serve de teste para "há estrutura de blocos aqui?". E o
subespaço do LDA é limitado a `n_classes - 1`, o que torna comparações entre
granularidades (25 x 125 classes) dependentes dessa ressalva.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import train_one  # noqa: E402

# §2: convenções do pipeline. Mudar qualquer uma invalida todas as tabelas.
SEED = 42
N_FOLDS = 5


def scaled_splits(
    blocks: list[np.ndarray],
    fit: np.ndarray,
    validation: np.ndarray | None,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Padroniza **por bloco**, ajustando o scaler só nas linhas de treino.

    Ajustar o `StandardScaler` na matriz inteira vazaria a média e o desvio do
    teste; por bloco em vez de global porque os blocos têm escalas muito
    diferentes e o kNN do grafo de similaridade roda neste mesmo espaço com peso
    igual por dimensão (§7.10).
    """
    train_parts, validation_parts, test_parts = [], [], []
    for block in blocks:
        scaler = StandardScaler()
        train_parts.append(scaler.fit_transform(block[fit]).astype(np.float32))
        if validation is not None:
            validation_parts.append(scaler.transform(block[validation]).astype(np.float32))
        test_parts.append(scaler.transform(block[test]).astype(np.float32))
    train_x = np.hstack(train_parts).astype(np.float32, copy=False)
    validation_x = (
        np.hstack(validation_parts).astype(np.float32, copy=False)
        if validation is not None else None
    )
    test_x = np.hstack(test_parts).astype(np.float32, copy=False)
    return train_x, validation_x, test_x


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """BAcc é a métrica principal; acurácia e F1 macro saem juntas (§2).

    O §7.12 mede por que as três têm de sair juntas: treinar com `class_weight`
    otimiza BAcc e sacrifica 7,30pp de acurácia de propósito.
    """
    return {
        "bacc": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def evaluate_lda(
    blocks: list[np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
) -> dict[str, object]:
    """Sonda linear, determinística e barata. É o instrumento das grades grandes
    (o Shapley de 2^12 máscaras x 5 folds do §5.9c), e o §5.9d mede que ela
    **ordena views** igual ao MLP (Spearman +1,000 contra o Shapley)."""
    train_x, _, test_x = scaled_splits(blocks, outer_train, None, outer_test)
    n_classes = len(np.unique(labels))
    model = LinearDiscriminantAnalysis(priors=np.ones(n_classes) / n_classes)
    model.fit(train_x, labels[outer_train])
    prediction = model.predict(test_x)
    return {
        **prediction_metrics(labels[outer_test], prediction),
        "validation_bacc": np.nan, "selected_k": np.nan,
        "selected_weights": "", "best_epoch": np.nan,
    }


def evaluate_mlp(
    blocks: list[np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    epochs: int,
    patience: int,
    label_smoothing: float = 0.0,
    return_probs: bool = False,
    model_factory=None,
) -> dict[str, object]:
    """O caminho de produção: 10% do treino externo vira validação, semente
    `SEED + fold`.

    **Não reimplementar este laço.** O §5.6 registra o caso em que uma
    reimplementação omitiu `label_smoothing`, usou `SEED+fold` onde o chamador
    passava `SEED`, e redefiniu o `BranchMLP` -- custou 0,47pp e o braço de
    controle deixou de reproduzir o valor conhecido.

    `return_probs=True` e `model_factory=None` são passthrough aditivos pro
    `train_one` (mesmo default, mesma assinatura) -- ver o docstring dele.
    """
    fit, validation = train_test_split(
        outer_train, test_size=0.1, stratify=labels[outer_train], random_state=SEED + fold
    )
    train_x, validation_x, test_x = scaled_splits(blocks, fit, validation, outer_test)
    assert validation_x is not None
    metrics, _ = train_one(
        train_x, labels[fit], validation_x, labels[validation], test_x, labels[outer_test],
        [block.shape[1] for block in blocks], len(np.unique(labels)),
        SEED + fold, epochs, patience, label_smoothing=label_smoothing,
        return_probs=return_probs, model_factory=model_factory,
    )
    return {
        **metrics, "selected_k": np.nan, "selected_weights": "",
    }
