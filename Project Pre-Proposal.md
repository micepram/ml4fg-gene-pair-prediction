
### **Predicting Unseen Multi-Gene Perturbation Responses with Attention-Based Aggregation**

**Name: Pramika Garg**
**UNI: pg2899**

**1. Biological question:**  
Can we better predict how a cell’s gene expression changes when two genes are perturbed together, especially for gene combinations that were not observed during training? The combined effect of two perturbations can be non-additive: activating genes A and B together may produce a response that cannot be predicted simply by adding their individual effects. I want to test whether explicitly modeling interactions between perturbations improves prediction of these unseen combinations.

**2. ML methods/models:**  
I will build on GEARS, a graph neural network that predicts post-perturbation gene expression from an unperturbed expression profile and a set of perturbed genes. GEARS represents each perturbation with a learned embedding and combines multiple perturbations using a **sum operation followed by an MLP**. My main extension will replace this fixed sum with a **Set Transformer**, which uses attention to learn how the effects of two perturbed genes interact while remaining independent of their ordering. I will compare the original GEARS sum aggregation, a simple mean aggregation baseline, and the attention-based model. The input will be the control-cell expression profile and identities of the perturbed genes; the output will be the predicted post-perturbation gene-expression vector. Performance on held-out gene pairs will be evaluated using mean squared error, Pearson correlation, and recovery of differentially expressed genes.

**3. Data:**  
I will use the Norman et al. Perturb-seq dataset (GSE133344), which contains approximately **110,000 single cells across 287 perturbation conditions**, including 132 gene pairs as well as the corresponding single-gene perturbations. GEARS already provides preprocessing and loading support for this dataset, which should keep the project focused on modeling rather than data cleaning. 

**4. Relevant paper:**  
Roohani, Y., Huang, K. & Leskovec, J. _Predicting transcriptional outcomes of novel multigene perturbations with GEARS._ **Nature Biotechnology 42**, 927–935 (2024). GEARS provides both the main baseline and public implementation for this project. 