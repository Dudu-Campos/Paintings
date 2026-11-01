# Beyond First-Order Regions: Efficient and Interpretable Painting Style Classification with Multigraph Image Representations

> Working manuscript, version 0.1. Numerical sources are frozen in
> `results_master.csv`. Citation markers and the pending Felzenszwalb MLP result
> must be resolved before submission.

## Abstract

Painting style classification is commonly approached with deep visual models,
whose accuracy depends on large-scale pretraining or costly end-to-end
optimization and whose decisions are difficult to relate to explicit visual
properties. We investigate a complementary regime: style recognition from
compact, interpretable representations that require neither visual pretraining
nor end-to-end feature learning. Our base representation segments each painting
with SLIC and summarizes four region graphs together with classical color and
texture descriptors and a Hilbert-ordered spatial signal. On 79,207 paintings
from 27 WikiArt styles, five-fold cross-validation yields a balanced accuracy
of 45.22% with a branch-fusion multilayer perceptron, compared with 42.34% for a
ResNet-50 trained from random initialization on a closely related 79,936-image
cohort. A fixed-quadrant multiscale extension reaches 48.33% without
pretraining. We further test two structural hypotheses. First, 63 statistics
of the line graph of the SLIC region-adjacency graph improve the SLIC-only MLP
from 38.49% to 42.03% and the full visual MLP from 45.36% to 46.60%; the latter
improvement is significant against a dimension-matched shuffled placebo.
Second, applying the same multigraph construction to an independent
Felzenszwalb segmentation produces residual gains after the line graph under
both LDA and XGBoost. Redundancy and importance analyses show that the gains
cannot be reduced to duplicated columns and identify texture, graph topology,
and color dispersion as the dominant signals. Pretrained ImageNet models retain
the highest absolute accuracy, but the proposed representations provide a
competitive and auditable operating point when pretraining, compute, or model
transparency is constrained.

## 1. Introduction

The style of a painting is expressed through several interacting visual
mechanisms. Color distributions and local texture matter, but so do spatial
composition, the organization of coherent regions, and the relations between
their boundaries. A classifier that compresses an image into a single global
embedding may learn these cues, yet it does not expose which structural
properties carry the decision. This distinction matters in art analysis, where
the scientific question is often not only whether a label can be predicted,
but also which visual regularities separate styles.

Current image classification systems largely inherit their representations
from large supervised or self-supervised corpora. This is a powerful regime,
but it entangles the target result with external data, substantial optimization
cost, and an encoder whose internal variables are not directly meaningful.
Training a convolutional network from scratch removes external pretraining but
does not remove the computational burden, and it can be statistically
inefficient when the target taxonomy is fine-grained and imbalanced.

We study an alternative based on explicit image representations. A painting is
first decomposed into perceptually coherent regions. The regions become nodes
with color, position, and area attributes, and several edge systems express
different notions of neighborhood. Compact distributional statistics then
describe the resulting graphs. This region-based representation is combined
with classical color and texture descriptors and a one-dimensional traversal
that preserves spatial locality. Every variable has a defined origin, and no
visual encoder is pretrained.

The central question is not whether handcrafted variables can replace every
pretrained model. Instead, we ask when structured representations remain
competitive and which extensions genuinely add information. We consider three
forms of diversity: fixed spatial scales, second-order relations between
region boundaries, and alternative image segmentations. Each extension is
tested on the same canonical cohort and, for the graph modules, against a
dimension-matched shuffled placebo.

Our contributions are:

1. an explicit 258-dimensional painting representation that combines a
   four-family region multigraph, classical color/texture descriptors, and a
   Hilbert-ordered spatial signal;
2. a multiscale formulation using the global image and four position-preserving
   quadrants, which improves balanced accuracy by 3.10 percentage points;
3. an interpretable line-graph representation of boundary junctions that adds
   consistent signal across LDA, XGBoost, and MLP classifiers;
4. evidence that SLIC and Felzenszwalb segmentations are complementary after
   controlling for dimensionality and linear redundancy;
5. an evaluation that separates no-pretraining methods from pretrained upper
   bounds and audits redundancy, feature importance, negative results, and
   computational cost.

The resulting picture is deliberately nuanced. ImageNet-pretrained encoders
achieve higher absolute accuracy, whereas the structured pipeline occupies a
stronger accuracy--cost--interpretability regime than the tested models learned
from scratch.

## 2. Related work

### 2.1 Computational analysis of artistic style

Computational art classification has used color histograms, texture statistics,
local descriptors, learned convolutional features, and, more recently,
transformer representations. The WikiArt collection has become a common test
bed for artist, genre, and style recognition, although its long-tailed labels,
metadata noise, and repeated artists complicate evaluation.

`[CITE: foundational WikiArt dataset and style-classification papers]`

Our setting differs from work whose primary objective is the maximum accuracy
available from a pretrained encoder. We treat pretrained CNNs as informative
upper bounds and focus on representations whose components can be traced back
to measurable image properties.

### 2.2 Texture and style representations

Texture is a strong cue for artistic style. Classical approaches use local
binary patterns, gray-level co-occurrence matrices, and color statistics.
Neural style representations instead correlate feature responses, most notably
through Gram matrices of convolutional activations. Such representations are
effective but inherit their feature space from the pretrained network.

`[CITE: LBP, Haralick/GLCM, Gatys neural style, compact neural texture representations]`

We include both classical Haralick/HOG features and a compact VGG Gram baseline.
This makes it possible to distinguish gains due to explicit structure from
gains due to pretrained neural texture features.

### 2.3 Region graphs and graph transformations

Region-adjacency graphs encode an image as coherent regions connected by shared
boundaries. Prior work has used superpixels, region graphs, graph kernels, and
graph neural networks for segmentation and image recognition. A line graph
transforms the edges of a graph into nodes and is therefore a natural language
for interactions between consecutive boundaries, but it has received less
attention as an interpretable aggregate representation for painting style.

`[CITE: SLIC, Felzenszwalb segmentation, RAG image recognition, graph-based art analysis, line graphs in vision]`

Our use of the line graph is not merely a second graph topology. Its attributes
measure continuity, junction geometry, and the relationship between regions on
opposite sides of adjacent boundaries.

### 2.4 Efficient and interpretable alternatives to pretrained vision

The growing cost of pretraining has motivated frozen encoders, parameter-
efficient adaptation, distillation, and explicit feature pipelines. These
approaches make different trade-offs: a frozen encoder reduces target-training
cost but still depends on external representation learning, whereas explicit
features reduce this dependency at the cost of a stronger inductive bias.

`[CITE: efficient transfer learning, interpretable visual features, compute-aware evaluation]`

We report these regimes separately rather than treating external pretraining as
a binary implementation detail.

## 3. Method

### 3.1 Overview

Let an RGB painting be denoted by \(I\). The global structured representation is

\[
\phi(I) = [\phi_{\text{graph}}(I),\; \phi_{\text{desc}}(I),\;
            \phi_{\text{Hilbert}}(I)],
\]

with dimensions 71, 130, and 57, respectively. The resulting 258-dimensional
vector is the canonical visual representation. All parameters are fixed before
the outer classification folds; labels are never used during feature
extraction.

We evaluate three optional transformations. The multiscale representation
concatenates position-specific copies of \(\phi\) from the global image and its
four quadrants. The line-graph block, \(\phi_L\), describes relations between
adjacent RAG boundaries. Finally, \(\phi_F\) applies the same 71-dimensional
multigraph summarization to a Felzenszwalb partition. These are modular views,
not additional labeled samples.

### 3.2 SLIC regions

Images are converted to RGB, downsampled with bilinear interpolation only when
their longer side exceeds 1,024 pixels, and converted to CIELAB. SLIC is run
with a requested budget of 2,048 segments, compactness 10, sigma 0, and
connectivity enforcement. The actual number of connected regions is recorded
for every image rather than assumed to equal the request.

For each region \(v\), we retain mean \(L,a,b\), normalized centroid \((x,y)\),
and area. The spatial region-adjacency graph (RAG) connects regions that meet
under a four-neighbor pixel boundary. Each edge stores the color distance and
centroid distance of its incident regions.

### 3.3 Four-family region multigraph

One SLIC partition generates four edge systems over the same region nodes:

- the spatial RAG;
- a 10-nearest-neighbor graph in standardized Lab+XY space;
- a Delaunay triangulation of region centroids;
- a minimum spanning tree of the RAG weighted by Lab distance.

For the RAG, 32 statistics summarize node count, edge count, density, region
area, color and spatial edge distances, degree distribution, spatial density,
and area-weighted color. The other three graphs contribute 13 edge-dependent
statistics each. Node statistics are stored once because their values are
shared. This yields \(32+3\times13=71\) dimensions.

### 3.4 Classical color and texture block

The image is resized to at most 128 pixels on its longer side. The
130-dimensional descriptor block comprises 90 Lab histogram bins, nine Lab
moments, an 18-bin uniform LBP histogram, and 13 Haralick/GLCM statistics. The
co-occurrence matrices use a one-pixel distance and four orientations.

### 3.5 Hilbert-ordered spatial block

The painting is resized to \(128\times128\), converted to Lab, and traversed by
a Hilbert space-filling curve. This maps each channel to a 16,384-sample signal
while preserving local spatial neighborhoods. For each channel, we compute the
mean and standard deviation within eight contiguous bands plus global mean and
standard deviation. Three pairwise channel correlations complete the
57-dimensional block.

### 3.6 Position-preserving multiscale views

The global image and the top-left, top-right, bottom-left, and bottom-right
quadrants are processed independently. Their positions are retained as
separate model branches. Each view contributes the same 258 dimensions, for a
total of 1,290. Quadrants are crops of the original image, not augmented
training examples, so the number of labels and outer-test observations remains
unchanged.

### 3.7 Boundary interactions through the line graph

Given a SLIC RAG \(G=(V,E)\), its line graph \(L(G)\) contains one node for each
region boundary \(e\in E\). Two line-graph nodes are adjacent when their
original boundaries meet at a shared region. We do not materialize a dense
line graph. Instead, incident edge lists enumerate boundary pairs at each
region, allowing the aggregate statistics to be computed directly.

The 63-dimensional block contains basic line-graph topology and summaries of:

- boundary contrast and spatial length;
- assortativity and change between consecutive boundaries;
- angles formed at shared region centroids;
- Lab and spatial distance between the two external regions;
- continuity of high-contrast boundaries;
- luminance reversals;
- endpoint area imbalance, orientation coherence, and spatial concentration.

These variables encode how boundaries continue, turn, or reverse around
regions--information not present in marginal RAG edge distributions.

### 3.8 A second segmentation view

Felzenszwalb segmentation is run with scale 50, sigma 0.8, minimum region size
50, and the same 1,024-pixel maximum side. Its regions pass through the same
RAG, kNN, Delaunay, MST, and 71-statistic pipeline used for SLIC. Consequently,
differences between the two blocks arise from the partition rather than from a
different feature definition.

### 3.9 Classifiers and fusion

We use LDA, XGBoost, and MLP to test whether a representation contributes under
linear, tree-based, and nonlinear decision functions. LDA uses median
imputation, standardization learned on the training fold, equal class priors,
and up to 26 discriminant components. XGBoost uses 100 depth-3 trees, learning
rate 0.1, row and column subsampling of 0.8, regularization, class-balanced
sample weights, and histogram tree construction.

Module evaluations use a four-hidden-layer MLP with widths 512, 512, 256, and
128, GELU activations, batch normalization, and dropout 0.35. AdamW optimization
uses learning rate \(3\times10^{-4}\), weight decay \(10^{-4}\), cosine decay,
class-weighted cross-entropy, batches of 4,096, at most 150 epochs, and patience
15.

The canonical and multiscale results use a branch-fusion MLP. Each
representation block first passes through a 128-unit encoder before the
embeddings are concatenated and processed by a 512--256--128 fusion head. A
paired ablation showed that branch fusion (45.85%) and the corresponding flat
MLP (45.78%) are effectively tied; architectural complexity is therefore not
the main source of performance.

## 4. Experimental protocol

### 4.1 Dataset and canonical cohort

We evaluate 27 style labels from WikiArt. The canonical cohort is the one-to-one
intersection of SLIC graphs, classical descriptors, Hilbert features, exported
RAGs, and labels, yielding 79,207 paintings. All joins use unique image stems,
and rows are sorted before fold construction. The metadata source contains
79,936 rows; 34 ambiguous duplicated stems are not allowed to create arbitrary
cross-modal joins.

`[CITE: WikiArt data source and license; describe exact 27-style selection]`

### 4.2 Cross-validation and metrics

We use five-fold stratified cross-validation with random seed 42. For neural
classifiers, 10% of each outer-training partition is reserved as an internal
stratified validation set. Early stopping observes only this internal set. The
outer test fold is used once for scoring.

The primary metric is balanced accuracy,

\[
\operatorname{BAcc}=\frac{1}{C}\sum_{c=1}^{C}
\frac{\operatorname{TP}_c}{\operatorname{TP}_c+\operatorname{FN}_c},
\]

which gives equal weight to all 27 styles. We report the mean and sample
standard deviation over five outer folds. Accuracy and macro-F1 are retained in
the released result files but are secondary.

### 4.3 Paired controls

To test whether an added block contains image-specific information rather than
merely increasing dimensionality or changing regularization, we permute all
rows of that block with one fixed seed while leaving the baseline block and
labels unchanged. Real and placebo additions have identical dimension and
marginal feature distributions. We compare fold-wise scores with paired
two-sided t-tests and report the five individual differences. Because five
folds provide limited inferential resolution, p-values support rather than
replace effect sizes and consistency.

For redundancy, each new feature is linearly regressed on individual baseline
features through squared Pearson correlation. We retain a conservative subset
whose maximum pairwise \(r^2\) is below 0.90 and repeat the module evaluation.

### 4.4 Baselines

No-pretraining baselines include Haralick+HOG, kNN, Random Forest, RBF-SVM, a
ResNet-50 trained from random initialization, and a graph neural network over
the raw SLIC RAG. Pretrained comparisons include frozen VGG-19 pooling features,
a compact VGG Gram representation, a frozen ResNet-50 embedding, and a fully
fine-tuned ResNet-50. Pretraining status and cohort size are always reported.

### 4.5 Computational measurements

The current timing registry records approximately 64 minutes for the canonical
visual pipeline and 681 minutes for five-fold ResNet-50 training from scratch.
The first value includes estimated pipeline components whereas the second is a
measured wall time. We therefore treat the current Pareto curve as provisional.
The final paper will repeat end-to-end measurements under fixed hardware,
thread counts, and CPU/GPU policy, reporting feature extraction separately from
classifier fitting.

## 5. Results

### 5.1 Visual classification without pretraining

Table 1 compares methods that do not inherit a pretrained visual encoder. The
canonical representation improves over the strongest classical control and the
ResNet-50 trained from scratch. Fixed spatial views provide a further 3.10
percentage points and form the strongest confirmed visual-only configuration.

**Table 1. No-pretraining results. Standard deviations are sample standard
deviations over five folds. The ResNet uses a slightly larger cohort.**

| Method | Images | BAcc |
|---|---:|---:|
| Chance | 79,207 | 3.70% |
| Haralick + HOG + MLP | 79,207 | 22.68% ± 0.70 |
| kNN on Visual-258 | 79,207 | 28.09% ± 0.54 |
| Random Forest on Visual-258 | 79,207 | 28.67% ± 0.56 |
| ResNet-50 from scratch | 79,936 | 42.34% ± 2.48 |
| RBF-SVM on Visual-258 | 79,207 | 43.18% ± 0.50 |
| Visual-258 + branch MLP | 79,207 | **45.22% ± 1.10** |
| Visual multiscale + branch MLP | 79,207 | **48.33% ± 1.28** |

The comparison with the scratch CNN is not paired, but its direction is clear:
the structured representation is 2.88 points higher in the single-view setting
and 5.99 points higher with fixed quadrants. The lower fold variance of the
multiscale representation also suggests that explicit spatial coverage reduces
sensitivity to a particular split.

### 5.2 Does the line graph add second-order information?

Table 2 gives the full-cohort paired results. The line graph improves SLIC in
all three classifiers. The effect persists after the 130 classical descriptors
and 57 Hilbert features are present, although the remaining margin is smaller.

**Table 2. Line-graph ablation on the canonical 79,207-image cohort.**

| Representation | LDA | XGBoost | MLP |
|---|---:|---:|---:|
| Line graph-63 alone | 27.02% ± 0.50 | 30.51% ± 0.51 | 34.98% ± 0.71 |
| SLIC-71 | 30.57% ± 0.72 | 34.79% ± 1.24 | 38.49% ± 0.59 |
| SLIC-71 + line-63 | **33.80% ± 0.49** | **36.80% ± 0.29** | **42.03% ± 0.48** |
| Visual-258 | 37.34% ± 0.64 | 39.65% ± 0.57 | 45.36% ± 1.24 |
| Visual-258 + line-63 | **38.92% ± 0.77** | **40.35% ± 0.85** | **46.60% ± 0.39** |

Relative to SLIC-71, the gains are +3.23, +2.01, and +3.54 points for LDA,
XGBoost, and MLP. Their paired p-values are 0.000585, 0.0103, and 0.000220,
respectively. Relative to Visual-258, the gains are +1.58, +0.70, and +1.24
points. The first two comparisons reach p<0.05; the MLP comparison has
p=0.0975 because the baseline varies more strongly across folds. Nevertheless,
the real line block exceeds the shuffled line placebo by 2.83 points in the
MLP (p=0.00329).

Forty-six of 63 line-graph features have no SLIC feature with \(r^2\ge0.90\).
Restricting the MLP to this subset still improves SLIC by 3.22 points. Thus,
the effect cannot be explained by exact copies of first-order RAG summaries.

### 5.3 Are different segmentations complementary?

Felzenszwalb is not uniformly better as a standalone segmenter. Its XGBoost
result, 34.84%, is essentially tied with the 34.79% SLIC result. The
combination, however, improves both the SLIC graph and the complete visual
representation.

**Table 3. Felzenszwalb ablation. MLP cells await the completed external run.**

| Representation | LDA | XGBoost | MLP |
|---|---:|---:|---:|
| Felzenszwalb-71 alone | 31.73% ± 0.45 | 34.84% ± 0.69 | pending |
| Visual-258 + Felzenszwalb-71 | 40.03% ± 0.75 | 40.52% ± 0.95 | pending |
| Visual-258 + line-63 | 38.92% ± 0.77 | 40.35% ± 0.85 | 46.60% ± 0.39 |
| Visual-258 + line-63 + Felzenszwalb-71 | **40.94% ± 0.73** | **41.26% ± 1.01** | pending |

After the line graph is already present, Felzenszwalb adds 2.02 points under
LDA (p=0.000258) and 0.91 points under XGBoost (p=0.0252). The XGBoost gain is
positive in all five folds and exceeds the shuffled Felzenszwalb placebo by
1.29 points (p=0.00428). Sixty-two of 71 Felzenszwalb features retain
\(r^2<0.90\) against Visual-258 + line, indicating that different partitions
expose genuinely different aggregate structures.

`[TODO: insert MLP Felzenszwalb values and revise the paragraph without changing other rows]`

### 5.4 Which representations and features matter?

The 258-dimensional visual vector is not dominated by widespread duplication.
After excluding two constant columns, the median pairwise \(r^2\) is 0.00477,
the 95th percentile is 0.3047, and only 24 pairs reach 0.99. The main
near-identities are repeated global Lab statistics and algebraically related
graph-size measures.

LDA structure coefficients allocate 41.21% of their mass to classical
descriptors, 36.20% to the SLIC multigraph, and 22.59% to Hilbert features.
XGBoost gain yields a similar 43.36%, 36.56%, and 20.08% allocation. Removing
and retraining confirms that all three representations contribute: the LDA
balanced-accuracy drops are 4.33, 3.96, and 1.16 points, respectively.

The highest-ranked variables agree on a substantive interpretation. Haralick
homogeneity and correlation, LBP bins, MST and kNN degree summaries, and the
dispersion of color distances dominate. The correlation between LDA structural
rank and XGBoost gain rank is moderate (Spearman 0.517), and 14 of the top 25
features overlap. Style information is therefore distributed across texture,
color organization, and region topology rather than captured by a single
feature family.

### 5.5 Comparison with pretrained representations

**Table 4. Pretrained visual comparisons. These methods answer a different
resource question from Table 1.**

| Method | Pretraining | Images | BAcc |
|---|---|---:|---:|
| VGG-19 conv5 GAP + MLP | ImageNet | 79,207 | 45.83% ± 1.35 |
| VGG-19 final GAP + MLP | ImageNet | 79,207 | 49.18% ± 0.88 |
| Compact VGG-19 Gram + MLP | ImageNet | 79,207 | 51.48% ± 0.88 |
| Frozen ResNet-50 + linear head | ImageNet | 79,936 | 55.46% ± 1.53 |
| Fine-tuned ResNet-50 | ImageNet | 79,936 | **62.72% ± 1.29** |

The compact Gram representation confirms the importance of second-order
texture statistics, but its advantage also depends on ImageNet features. Full
fine-tuning remains the best accuracy-oriented solution. Our result instead
defines a no-pretraining operating point with explicit, inspectable variables.

### 5.6 Accuracy and cost

The recorded timings place the canonical visual method at approximately 64
minutes and the five-fold scratch ResNet-50 at approximately 681 minutes. This
suggests that the structured method can dominate the scratch CNN in both
accuracy and wall time. Because one timing is partly estimated and the current
experiments were accumulated under more than one hardware policy, we do not
yet present the ratio as a definitive benchmark. A controlled rerun will
separate image preprocessing, feature extraction, and classifier training.

The incremental line-graph computation is already demonstrably small: once
the SLIC RAGs exist, all 79,207 feature rows were produced in approximately
274 seconds with no failures. Felzenszwalb is the more expensive extension and
must be included explicitly in the final Pareto analysis.

## 6. Controlled textual extension

We also ask whether an independently generated description contributes beyond
the numeric image representation. To minimize direct target leakage, we use a
public CATR captioner trained on human MS COCO captions and never fine-tuned on
WikiArt. A sparse text classifier produces 27 out-of-fold decision scores;
vocabulary and classifier fitting are nested inside each outer training fold.

CATR text alone reaches 19.72% OOF balanced accuracy. Adding its scores to the
Numeric-274 representation improves OOF balanced accuracy by 1.63 points for
LDA, 1.58 for XGBoost, and 0.79 for the branch MLP. A lexical audit of 12,210
captions finds no style or artist names and only three exact title coincidences.

This analysis is secondary for an important reason. Numeric-274 contains the
258 visual variables plus 16 sentiment variables extracted from a legacy
captioner with Qwen/ArtEmis lineage. The incremental CATR scores are produced
without WikiArt supervision, but the absolute fused pipeline is not entirely
free of the earlier text lineage. We therefore use this experiment only as
evidence that object-level caption content is complementary and do not include
its absolute accuracy in the main visual claim.

## 7. Negative and diagnostic experiments

Several plausible extensions did not justify their cost or complexity. A GNN
over the raw SLIC RAG fused with the 71 aggregate graph features reaches 42.15%
and takes 11.43 hours, remaining below the compact visual vector. A second
line-graph transformation (the square graph) saturates after the first line
graph in a 3,000-image study. Real-ESRGAN gives only a 0.86-point XGBoost gain
over the original representation in the same pilot (p=0.116) and would require
roughly 30 hours for the full corpus. Big-LaMa outpainting yields less than one
point beyond a blank-canvas control. More elaborate early- and late-fusion
architectures are tied with simple concatenation.

These results constrain the interpretation of the positive findings. The gain
does not follow automatically from deeper models, higher image resolution,
more generated pixels, or more graph transformations. It appears when the
additional view captures a specific missing relation at moderate complexity.

## 8. Discussion

### 8.1 Why explicit structure remains competitive

Painting styles change the statistics of edges, texture, color, and
composition. The proposed representation injects these invariances directly:
regions suppress pixel-level noise, graph summaries ignore arbitrary region
ordering, and Hilbert traversal retains coarse spatial continuity. This makes
the classifier's learning problem smaller than learning both the visual basis
and the class boundary from scratch.

### 8.2 What the line graph contributes

A RAG edge distribution can say that a painting contains strong boundaries,
but not whether strong boundaries continue through a region, intersect at
particular angles, or separate external regions with similar colors. The line
graph makes these relationships first-class observations. Its stronger gain
over SLIC than over the full 258-dimensional vector is expected: classical
texture and Hilbert summaries already capture part of the missing context.

### 8.3 Why two segmentations help

SLIC favors a controlled budget of compact regions, whereas Felzenszwalb adapts
region size to image evidence. Their standalone accuracies can be similar while
their errors and summary statistics differ. The large nonredundant fraction and
the residual gain after the line graph show that segmentation is not merely a
preprocessing choice; it defines a view of image organization.

### 8.4 Accuracy, pretraining, and interpretability are separate axes

The fine-tuned ResNet-50 is the appropriate choice when maximum balanced
accuracy is the only objective and ImageNet pretraining is acceptable. The
structured representation serves a different scientific and operational need:
its features can be named, audited, permuted, and connected to color, texture,
or topology. A fair comparison should therefore expose pretraining and compute
rather than collapse all methods into one ranking.

## 9. Limitations and threats to validity

First, the primary split is stratified by style but not grouped by artist.
Many WikiArt artists are strongly associated with one style, and 1,029 artists
cross the available partitions. An artist-disjoint evaluation is required to
quantify how much performance reflects artist-specific signatures.

Second, the image corpus may contain exact or near duplicates. Joining on
unique stems removes ambiguous identifiers but does not prove visual
independence across folds. A perceptual-hash and embedding-based duplicate
audit is planned.

Third, the CNN controls use 79,936 images while the canonical structured cohort
contains 79,207. This difference is small but prevents paired significance
tests. We label it rather than interpreting the differences as exact matched
effects.

Fourth, module significance is computed from five correlated training/test
partitions. The shuffled controls and fold-wise consistency strengthen the
evidence, but confidence intervals based on more repeated grouped splits would
be preferable. Multiple exploratory hypotheses were tested over the course of
the project; the main article therefore distinguishes confirmatory full-cohort
results from pilots and negative searches.

Fifth, the current timing comparison is not yet a controlled systems benchmark.
Hardware, thread counts, CPU/GPU policy, data caching, and the boundary between
one-time extraction and per-fold training must be fixed before making a precise
cost-ratio claim.

Finally, the approach intentionally compresses region graphs into aggregate
statistics. This provides auditability and speed but discards localized motifs
and may cap performance relative to large pretrained encoders. The negative GNN
result evaluates one architecture, not every possible graph-learning method.

## 10. Conclusion

Explicit image structure remains a viable basis for painting style
classification. A 258-dimensional combination of region multigraph, classical
descriptor, and Hilbert features outperforms the tested ResNet-50 trained from
scratch, while fixed spatial views improve it further. Line-graph statistics
show that boundary interactions contain information beyond first-order region
summaries, and an independent Felzenszwalb view demonstrates that different
partitions expose complementary structure. Feature audits connect these gains
to texture, topology, and color dispersion rather than opaque latent units.

Pretrained CNNs still provide the highest accuracy. The contribution of this
work is therefore not a universal replacement for transfer learning, but a
compact, interpretable, and compute-aware alternative whose assumptions and
failure modes can be inspected directly.

## Appendix plan

### A. Per-fold results and paired tests

Full line-graph and Felzenszwalb fold tables, shuffled controls, nonredundant
subsets, and exact p-values.

### B. Feature definitions

Complete dictionary of the 71 SLIC/Felzenszwalb features, 63 line-graph
features, 130 descriptors, and 57 Hilbert features.

### C. Confusion and per-style analysis

Normalized confusion matrix, recall by style, and dominant confusion pairs.

### D. Negative experiments

GNN, square graph, super-resolution, outpainting, fusion architectures, and
captioner failure analysis.

### E. Text leakage audits

CATR, ArtEmis-derived captioner, and direct-Qwen lexical audits, including
seen/held-out comparisons and scrubbed-text controls.

### F. Reproducibility

Environment versions, hardware, thread controls, runtime decomposition, seeds,
dataset checksums, and paths from `results_master.csv`.

