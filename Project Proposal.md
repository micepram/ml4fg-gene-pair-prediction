# Predicting Unseen Gene-Pair Responses with Biological Gene Embeddings and Set Attention

**Name:** Pramika Garg  
**UNI:** pg2899

## Biological question and motivation

Can biological gene embeddings and attention improve prediction of transcriptional responses to gene pairs not perturbed together during training? Joint activation can produce suppression or synergistic differentiation [1]. Predicting these non-additive responses could help prioritize experimental combinations.

I will extend GEARS [2] to distinguish gains from better gene representations and better aggregation. The professor-recommended MORPH preprint [3] compares expression-derived, Geneformer, GenePT, and DepMap priors, reporting particularly strong generalization with DepMap dependency profiles. GenePT [4] encodes gene descriptions, while scGenePT [5] demonstrates the value of textual knowledge for perturbation prediction. These findings motivate testing functional and text-derived priors alongside attention.

## Data and representation

I will use the Norman et al. K562 CRISPR-activation Perturb-seq dataset (**GSE133344**) through the [GEARS loader](https://github.com/snap-stanford/GEARS). The original screen contains approximately 110,000 cells across 287 conditions, including 132 gene pairs; GEARS analyzes 131 pairs after processing [1,2]. I will use its processed AnnData (`.h5ad`) expression matrix and labels, verify normalization/log transformation, and document retained sample sizes.

Inputs are a control-cell expression vector and an unordered set of one or two perturbed gene identities; outputs are predicted expression vectors. Cells are unpaired, so evaluation will use condition-level means. External features will be published GenePT NCBI/UniProt embeddings and DepMap dependency profiles across cell lines, from the authors’ [GenePT](https://github.com/yiqunchen/GenePT) and [MORPH](https://github.com/uhlerlab/MORPH) resources. I will align identifiers, report coverage, and encode missing priors with zeros plus an indicator. DepMap features will exclude K562; their loss-of-function viability signal may not transfer to CRISPR activation.

## Proposed method

GEARS sums graph-informed perturbation embeddings and applies an MLP [2], which can already represent interactions. I will test whether a small Set Transformer [6] improves generalization, using self-attention without positional encodings followed by attention pooling to preserve gene-order invariance.

A learned projection will combine each GEARS perturbation embedding with a frozen external prior before aggregation. A **two-by-three comparison** will cross sum versus attention with native GEARS embeddings, GEARS plus GenePT, and GEARS plus DepMap. Graphs, decoder, hidden dimension, and objective will remain fixed. A capacity-matched MLP and shuffled-prior ablation will test whether gains reflect additional parameters or biological information.

## Evaluation and baselines

All models will share perturbation-level splits, keeping each condition’s cells together. The primary task holds out pairs while retaining their singles in training. A secondary GEARS-style split will report pairs containing zero, one, or two genes unseen in any training perturbation. Training will include some pairs. Expression-derived graphs and fitted preprocessing will use training data only; model selection will use validation conditions.

Baselines will include GEARS, no-change prediction, the mean training response, and, where training singles exist, the additive prediction **μ̂AB = μA + μB − μcontrol**. Metrics will be MSE, Pearson correlation, and R² of mean expression changes relative to control, over all genes and the top 20 observed differentially expressed genes selected only for evaluation. For pairs with training singles, I will assess the interaction residual **rAB = μAB − μA − μB + μcontrol**, including strongly non-additive pairs. Conditions will receive equal weight, with three seeds and paired bootstrap confidence intervals over held-out pairs.

## Interim milestones and scope

By the interim report, I plan to audit the data and embedding coverage, visualize response magnitudes/non-additivity, establish splits, reproduce baselines, and run one embedding-augmented pilot. The final phase will implement attention and complete comparisons. The contribution is a GEARS extension evaluating whether biological priors, attention, or their combination improve unseen-pair prediction within this screen.

## References

1. Norman, T. M. et al. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science* **365**, 786–793 (2019). [Paper](https://doi.org/10.1126/science.aax4438).
2. Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology* **42**, 927–935 (2024). [Paper](https://doi.org/10.1038/s41587-023-01905-6).
3. He, C., Zhang, J., Dahleh, M. A. & Uhler, C. MORPH Predicts the Single-Cell Outcome of Genetic Perturbations Across Conditions and Data Modalities. *bioRxiv* (2025). [Professor-linked revision, v2](https://www.biorxiv.org/content/10.1101/2025.06.27.661992v2); [open full text of v1, including the prior comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC12236822/).
4. Chen, Y. & Zou, J. Simple and effective embedding model for single-cell biology built from ChatGPT. *Nature Biomedical Engineering* **9**, 483–493 (2025). [Paper](https://doi.org/10.1038/s41551-024-01284-6).
5. Istrate, A.-M., Li, D. & Karaletsos, T. scGenePT: Is language all you need for modeling single-cell perturbations? *bioRxiv* (2024). [Paper](https://doi.org/10.1101/2024.10.23.619972).
6. Lee, J. et al. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. *Proceedings of ICML*, PMLR **97**, 3744–3753 (2019). [Paper](https://proceedings.mlr.press/v97/lee19d.html).
