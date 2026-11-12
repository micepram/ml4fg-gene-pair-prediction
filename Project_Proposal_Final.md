# Predicting Gene-Pair Responses Beyond Additivity with Biological Embeddings and Set Attention

**Name:** Pramika Garg  
**UNI:** pg2899

## Biological question and motivation

Designing combination therapeutics requires identifying targets whose joint action produces a useful cellular response. Norman et al. demonstrated reinforcing and suppressive interactions, including gene combinations that promote differentiation [1]. Can biological embeddings and attention predict unseen gene-pair responses better than adding individual effects? This project tests methods for prioritizing target combinations; expression changes in one cancer cell line cannot establish drug efficacy.

Ahlmann-Eltze, Huber and Anders found that additivity outperformed the tested deep-learning models [2]. This supports a demanding comparator, without ruling out improvements from other representations. Newer work questions metric sensitivity but still finds additivity difficult to surpass on Norman [3]. I will assess expression error and interaction-specific signal.

## Data and proposed method

I will use the benchmark's processed Norman K562 Perturb-seq AnnData (`.h5ad`): 100 single-gene conditions, 124 pairs and 19,264 measured genes [2]. This differs from the previously cited 131-pair GEARS release. I will check retained sample sizes, normalization and log transformation, and build expression-derived graphs from training conditions only.

Inputs will be control-cell expression vectors and unordered perturbed-gene sets; outputs will be predicted expression vectors. Cells are unpaired, so evaluation will compare condition means. I will combine GEARS embeddings [4] with frozen GenePT gene-description embeddings [5] or DepMap dependency profiles, motivated by MORPH [6]. Missing features will receive zeros and an indicator. DepMap features will exclude K562; their loss-of-function viability signal may not transfer to gene activation.

Six configurations will cross sum versus Set Transformer aggregation [7] with native GEARS, GEARS plus GenePT, and GEARS plus DepMap embeddings. Attention will preserve gene-order invariance without positional encodings. Decoder, hidden dimension and training budget will remain fixed. GEARS already represents nonlinear effects; improvements are a hypothesis.

## Evaluation

I will use **five prespecified random pair splits**, with 50% training, 25% validation and 25% test pairs: 62/31/31 for 124 pairs. All singles and controls will remain available for training; each pair's cells will stay together. Configurations will share splits and two optimizer seeds per split. Hyperparameters and checkpoints will use validation pairs only. The task is unseen combinations of individually observed genes.

The additive prediction, computed on the same log-expression scale, is

$$
\widehat{\mu}^{\mathrm{add}}_{AB}=\mu_A+\mu_B-\mu_{\mathrm{ctrl}}.
$$

Other baselines will be unmodified GEARS, no change, and the mean training-condition response. Primary evaluation will use MSE on the 1,000 genes most highly expressed in training controls, matching [2]. For each held-out pair and split, I will report

$$
D_{p,s}=\mathrm{MSE}_{\mathrm{add},p,s}-\mathrm{MSE}_{\mathrm{model},p,s},
$$

averaging errors across optimizer seeds; positive values favor the model. I will plot pair differences and split means, averaging repeated held-out results within each pair before equal-weight summaries and 95% pair-bootstrap intervals conditional on fitted models. Repeated appearances are not independent observations. Attention plus DepMap versus additivity is the primary comparison; other configurations examine embedding and aggregation effects.

Secondary metrics will assess control-relative Pearson correlation, MSE on the top 20 differentially expressed genes selected only for scoring, and correlation between predicted and observed interaction residuals $r_{AB}=\mu_{AB}-\widehat{\mu}^{\mathrm{add}}_{AB}$. Independent cell halves within control, single and pair conditions will estimate residual repeatability. These checks will not replace the primary metric after results are known.

## Interim milestones and expected contribution

For the interim report, I will check data and embedding coverage, generate splits, reproduce additive and GEARS baselines, plot pair errors, and run one embedding extension. I will then complete the comparisons and repeatability checks. The contribution is a GEARS extension tested against additivity. Small or inconsistent differences will remain inconclusive, rather than evidence that gene interactions are absent.

## References

1. Norman, T. M. et al. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science* **365**, 786–793 (2019). [Paper](https://doi.org/10.1126/science.aax4438).
2. Ahlmann-Eltze, C., Huber, W. & Anders, S. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nature Methods* **22**, 1657–1661 (2025). [Paper](https://doi.org/10.1038/s41592-025-02772-6); [code and data](https://zenodo.org/records/16092690).
3. Miller, H. E. et al. Deep learning perturbation models can outperform baselines on calibrated metrics. *Nature Biotechnology* (2026). [Paper](https://doi.org/10.1038/s41587-026-03307-w).
4. Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology* **42**, 927–935 (2024). [Paper](https://doi.org/10.1038/s41587-023-01905-6).
5. Chen, Y. & Zou, J. Simple and effective embedding model for single-cell biology built from ChatGPT. *Nature Biomedical Engineering* **9**, 483–493 (2025). [Paper](https://doi.org/10.1038/s41551-024-01284-6).
6. He, C., Zhang, J., Dahleh, M. A. & Uhler, C. MORPH Predicts the Single-Cell Outcome of Genetic Perturbations Across Conditions and Data Modalities. *bioRxiv* (2025). [Professor-linked v2](https://www.biorxiv.org/content/10.1101/2025.06.27.661992v2); [accessible v1 prior comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC12236822/).
7. Lee, J. et al. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. *Proceedings of ICML*, PMLR **97**, 3744–3753 (2019). [Paper](https://proceedings.mlr.press/v97/lee19d.html).
