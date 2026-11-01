CONFERENCIA contra os valores registrados no §4.1f (tolerancia 0,01pp)
  ok  rijksdraw    sem  registrado +3.83  medido +3.83  (0.003)
  ok  rijksdraw    com  registrado +3.83  medido +3.83  (0.001)
  ok  nct_crc      sem  registrado +0.92  medido +0.92  (0.003)
  ok  nct_crc      com  registrado +1.06  medido +1.06  (0.003)
  ok  multitask    sem  registrado +3.99  medido +3.99  (0.003)
  ok  multitask    com  registrado +4.58  medido +4.58  (0.003)
  ok  oilpainting  sem  registrado +1.47  medido +1.47  (0.002)
  ok  oilpainting  com  registrado +2.08  medido +2.08  (0.002)
  ok  places_lt    sem  registrado -0.28  medido -0.28  (0.002)
  ok  places_lt    com  registrado +1.42  medido +1.42  (0.005)
  ok  minc2500     sem  registrado -1.69  medido -1.69  (0.004)
  ok  minc2500     com  registrado -0.40  medido -0.40  (0.002)
  ok  food101      sem  registrado -0.88  medido -0.88  (0.003)
  ok  food101      com  registrado +0.29  medido +0.29  (0.001)
  maior divergencia: 0.0049pp

unidades por celula: [np.float64(10.0)] (10 = 2 modelos x 5 sementes)

==============================================================================
TABELA PRINCIPAL -- fusao de decisao com peso FIXO w=0,5, 10 unidades por celula
==============================================================================

| corpus | cond. | nosso | backbone | gap | fusao | Δ | p | unid. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rijksdraw | sem texto | 56.67 | 58.63 | 2.0 | 62.46 | **+3.83 ± 0.98** | 6.1e-07 | 10/10 |
| rijksdraw | **com texto** | 57.03 | 58.63 | 1.6 | 62.46 | **+3.83 ± 1.03** | 9.3e-07 | 10/10 |
| nct_crc | sem texto | 87.46 | 94.17 | 6.7 | 95.10 | **+0.92 ± 0.67** | 0.0018 | 10/10 |
| nct_crc | **com texto** | 88.27 | 94.17 | 5.9 | 95.24 | **+1.06 ± 0.65** | 0.00058 | 10/10 |
| multitask | sem texto | 38.25 | 48.83 | 10.6 | 52.81 | **+3.99 ± 0.36** | 5.7e-11 | 10/10 |
| multitask | **com texto** | 40.21 | 48.83 | 8.6 | 53.41 | **+4.58 ± 0.58** | 1.3e-09 | 10/10 |
| oilpainting | sem texto | 43.61 | 62.35 | 18.7 | 63.82 | **+1.47 ± 0.37** | 4.8e-07 | 10/10 |
| oilpainting | **com texto** | 47.77 | 62.35 | 14.6 | 64.44 | **+2.08 ± 0.30** | 3.8e-09 | 10/10 |
| places_lt | sem texto | 14.07 | 33.84 | 19.8 | 33.56 | **-0.28 ± 0.64** | 0.21 | 3/10 |
| places_lt | **com texto** | 22.77 | 33.84 | 11.1 | 35.26 | **+1.42 ± 0.65** | 7.3e-05 | 10/10 |
| minc2500 | sem texto | 54.34 | 81.47 | 27.1 | 79.78 | **-1.69 ± 1.01** | 0.00048 | 1/10 |
| minc2500 | **com texto** | 63.69 | 81.47 | 17.8 | 81.07 | **-0.40 ± 1.13** | 0.29 | 4/10 |
| food101 | sem texto | 38.58 | 86.93 | 48.3 | 86.04 | **-0.88 ± 0.73** | 0.0041 | 0/10 |
| food101 | **com texto** | 58.36 | 86.93 | 28.6 | 87.22 | **+0.29 ± 0.54** | 0.12 | 8/10 |


| corpus | Δ sem texto | Δ com texto | gap sem → com |
|---|---:|---:|---:|
| rijksdraw | +3.83 (10/10) | +3.83 (10/10) | 2.0 → 1.6 |
| nct_crc | +0.92 (10/10) | +1.06 (10/10) | 6.7 → 5.9 |
| multitask | +3.99 (10/10) | +4.58 (10/10) | 10.6 → 8.6 |
| oilpainting | +1.47 (10/10) | +2.08 (10/10) | 18.7 → 14.6 |
| places_lt | -0.28 (3/10) | +1.42 (10/10) | 19.8 → 11.1 |
| minc2500 | -1.69 (1/10) | -0.40 (4/10) | 27.1 → 17.8 |
| food101 | -0.88 (0/10) | +0.29 (8/10) | 48.3 → 28.6 |


teste balanceado -> Acc = BAcc: maior divergencia 0.0000pp em ['food101', 'minc2500', 'places_lt']

Δ DA FUSAO NAS TRES METRICAS (w=0,5, 10 unidades por celula)

| corpus | cond. | Δ BAcc | Δ Acc | Δ macro-F1 | sinais |
|---|---|---:|---:|---:|---|
| rijksdraw | sem texto | +3.83 | +2.91 | +2.71 | concordam |
| rijksdraw | com texto | +3.83 | +2.95 | +2.84 | concordam |
| nct_crc | sem texto | +0.92 | +0.52 | +0.86 | concordam |
| nct_crc | com texto | +1.06 | +0.64 | +0.99 | concordam |
| multitask | sem texto | +3.99 | -0.15 | +0.97 | **DISCORDAM** |
| multitask | com texto | +4.58 | +0.98 | +2.08 | concordam |
| oilpainting | sem texto | +1.47 | +0.65 | +0.74 | concordam |
| oilpainting | com texto | +2.08 | +1.23 | +1.46 | concordam |
| places_lt | sem texto | -0.28 | -0.28 | -0.57 | concordam |
| places_lt | com texto | +1.42 | +1.42 | +0.94 | concordam |
| minc2500 | sem texto | -1.69 | -1.69 | -1.70 | concordam |
| minc2500 | com texto | -0.40 | -0.40 | -0.43 | concordam |
| food101 | sem texto | -0.88 | -0.88 | -0.89 | concordam |
| food101 | com texto | +0.29 | +0.29 | +0.30 | concordam |

celulas em que as tres metricas DISCORDAM de sinal: 1 de 14  [('multitask', 'sem')]

NIVEL absoluto da fusao nas tres metricas (media das 10 unidades)

| corpus | cond. | BAcc | Acc | macro-F1 |
|---|---|---:|---:|---:|
| rijksdraw | sem texto | 62.46 | 62.54 | 60.85 |
| rijksdraw | com texto | 62.46 | 62.57 | 60.99 |
| nct_crc | sem texto | 95.10 | 95.30 | 94.77 |
| nct_crc | com texto | 95.24 | 95.42 | 94.91 |
| multitask | sem texto | 52.81 | 33.59 | 35.82 |
| multitask | com texto | 53.41 | 34.73 | 36.93 |
| oilpainting | sem texto | 63.82 | 66.17 | 61.34 |
| oilpainting | com texto | 64.44 | 66.74 | 62.06 |
| places_lt | sem texto | 33.56 | 33.56 | 32.15 |
| places_lt | com texto | 35.26 | 35.26 | 33.66 |
| minc2500 | sem texto | 79.78 | 79.78 | 79.71 |
| minc2500 | com texto | 81.07 | 81.07 | 80.98 |
| food101 | sem texto | 86.04 | 86.04 | 86.00 |
| food101 | com texto | 87.22 | 87.22 | 87.19 |


O QUE O TEXTO MUDA NA FUSAO -- pareado por unidade (mesmo corpus/modelo/semente)

| corpus | Δ(com) − Δ(sem) | p | unid. a favor |
|---|---:|---:|---:|
| rijksdraw | -0.00 ± 0.18 | 0.97 | 5/10 |
| nct_crc | +0.14 ± 0.23 | 0.092 | 9/10 |
| multitask | +0.60 ± 0.50 | 0.0044 | 10/10 |
| oilpainting | +0.61 ± 0.31 | 0.00015 | 10/10 |
| places_lt | +1.69 ± 0.60 | 9.5e-06 | 10/10 |
| minc2500 | +1.29 ± 1.49 | 0.023 | 8/10 |
| food101 | +1.17 ± 0.30 | 5.2e-07 | 10/10 |

CELULAS QUE NAO SE SEPARAM DE ZERO (p > 0,05) -- a legenda tem de dize-lo,
nos dois sentidos: o negativo que sobra e os positivos pequenos.
  places_lt    sem texto   Δ -0.28  p=0.21  3/10
  minc2500     com texto   Δ -0.40  p=0.29  4/10
  food101      com texto   Δ +0.29  p=0.12  8/10

sem texto  corpora positivos 4/7 | unidades positivas 44/70 | faixa -1.69 a +3.99

com texto  corpora positivos 6/7 | unidades positivas 62/70 | faixa -0.40 a +4.58

escritos: results/fusion_main_table{,_by_model,_units}.csv
