# Research notes for the revised proposal

These notes explain the evidence and design choices behind `Project_Proposal_Final.md`. They are not part of the proposal submission. No model training or independent reproduction of the published results was performed during this revision.

## Is the professor's recommended paper valid and relevant?

Yes, within its tested setting: unseen combinations with singles available. The additive comparator uses permitted information. Five random partitions and several expression/interaction analyses make the study directly relevant. [Published article](https://doi.org/10.1038/s41592-025-02772-6), [full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12328236/fullTextXML).

The result concerns tested implementations, not every possible architecture. Limited dataset coverage, cancer cell lines and perturbation quality constrain generalization. It neither invalidates this extension nor establishes that attention will help.

The approximately 96% additivity figure concerns gene-by-pair expression entries, not whole gene pairs or a universal biological rate. Sparse deviations may still matter, motivating interaction-specific evaluation. [Fig. 1 and methods](https://doi.org/10.1038/s41592-025-02772-6).

## What checking the code changed

The publication-version preparation script downloads the scFoundation-processed Norman file and creates train/test/validation condition lists. The paired-condition split is 50% training, 25% test and 25% validation. The analysis notebook includes both labels `test` and `val` in its main double-perturbation summaries, accounting for the reported evaluation on the remaining half. Thus, the script's validation label and the paper's broad held-out description should not be treated as an untouched final test protocol without inspecting how each model uses validation data. [Preparation script, publication branch](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/publication/benchmark/src/prepare_perturbation_data.py), [analysis notebook](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/publication/notebooks/double_perturbation_analysis.Rmd).

This is a reason to keep validation and test results separate in the new project. It is not sufficient evidence to dismiss the published additive-baseline result. If validation-guided checkpoint selection affected pooled scores, the likely advantage would be to the fitted neural models, not to an additive prediction that never learns from pair responses. Determining its actual effect would require reproducing the exact environments and model versions. I inspected source code, not the archived prediction arrays or a rerun.

The additive implementation sums single-condition changes from the control mean in the processed expression space. It does not sum raw cell counts or compare an expression-space prediction against a different normalization. In this all-singles-observed task, its use of the single-condition data is legitimate. That baseline would not be directly available for an entirely unseen gene whose single-perturbation response was withheld. [Additive implementation](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/publication/benchmark/src/run_additive_model.py).

The processed release is also important: the recommended benchmark reports 124 pairs, whereas the earlier proposal used the 131-pair GEARS release. Those are different processing choices, not interchangeable sample sizes. The revision deliberately uses the benchmark release and requires reporting the actual retained counts before modeling. Published numbers are not presented as counts independently verified by loading the dataset. [Data preparation and download source](https://github.com/const-ae/linear_perturbation_prediction-Paper/blob/publication/benchmark/src/prepare_perturbation_data.py).

## Does newer evidence contradict the recommendation?

Miller et al., published on 1 October 2026, argue that some common metrics can miss useful perturbation-specific predictions. Their analyses include positive controls and metrics that emphasize affected genes. This weakens a universal interpretation that deep learning cannot predict perturbation effects. It does not remove the need for the additive comparator: their Norman results also describe it as difficult to surpass and caution against using Norman alone to establish broad ability to model non-additive responses. [Published study](https://doi.org/10.1038/s41587-026-03307-w).

The revision therefore fixes the primary metric and gene set before looking at test results, adds secondary checks for interaction signal and measurement repeatability, and limits conclusions to this screen. A favorable result on a secondary metric will be reported as such, rather than used to replace an unfavorable primary result. Another dataset would be needed for a broad generalization claim; it is outside the core course-project scope.

## Why the revised statistics are more defensible

- **Pair partitions and optimizer seeds answer different questions.** The former vary which biological combinations are learned; the latter measure training randomness on a fixed partition. Five partitions with two optimizer seeds give 60 neural fits across six configurations.
- **Pairing removes an avoidable source of comparison noise.** Each model and the additive baseline are scored against the same held-out pair, expression scale and genes. A positive additive-minus-model error difference means an improvement for that pair.
- **Repetition does not create new biology.** A pair tested in several partitions is still one pair. Averaging its held-out results before a pair bootstrap prevents counting those appearances as independent samples. The intervals are conditional summaries for fitted models; split-to-split variation is shown separately. Shared constituent genes and overlapping training sets still limit independence and generalization.
- **The primary comparison is chosen in advance.** Attention plus DepMap versus additivity is the main contrast. The remaining configurations help explain changes. A small advantage for whichever model happens to win on the test set will not establish superiority.
- **Residual error alone is not a second independent success criterion.** Subtracting the same additive vector from prediction and observation leaves their squared error unchanged. Residual correlation and repeatability can diagnose interaction signal, but relabeling the same MSE as interaction MSE provides no extra evidence. The additive model's residual is identically zero, so its residual Pearson correlation is undefined and must not be reported as zero.
- **Measurement noise needs its own check.** Independently splitting cells within controls, singles and pairs avoids making two residual estimates share the same noisy single/control means. These cell halves are diagnostic evaluation data, never inputs for fitting or selecting the prediction model. They estimate within-screen repeatability, not independent biological replication.

## Motivation and practical limits

The connection to designed combination therapeutics is a motivation for target prioritization. Norman is a gene-activation expression screen, not a drug-dose, toxicity, tumor-response or clinical-outcome study. Consequently, the proposal cannot establish that a predicted interaction makes an effective treatment. Similarly, MORPH's DepMap result motivates an embedding experiment; it does not establish that a GEARS extension will benefit from that prior or that knockout viability profiles necessarily transfer to activation. [Norman study](https://doi.org/10.1126/science.aax4438), [MORPH accessible full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12236822/).

The project remains a defensible course project because it extends model code and tests a specific hypothesis against a strong comparator. Its likely difficulty is detecting a stable gain from a small number of distinct combinations. A well-supported negative or inconclusive result would still answer the stated methodological question. It would not demonstrate absence of genetic interactions or failure of the whole approach in other datasets.
