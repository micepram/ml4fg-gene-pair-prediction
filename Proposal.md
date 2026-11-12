**

## Proposal Guidelines

I recommend writing proposals and final reports in Latex but this is not a requirement. Overleaf is easiest and has excellent collaboration/track changes for team work. In Google Docs it’s basically impossible to keep figures good quality: it doesn’t support vector graphics and forcibly reduces bitmap resolution (with no warning!) 

  

The proposal should be between .5 * (1+N) and .5 * (2+N) pages for N team members. Your proposal should include:

1. What biological question will your method answer/enable answering? 
    
2. What ML method(s)/model(s) will you try using/developing? What ML method(s)/model(s) will you try using/developing? If using a supervised model what will the inputs/outputs be and how will they be represented? How will you assess performance [hint: straight accuracy is difficult to interpret because it depends on class imbalance, AUROC or even better auPR are preferable, or percent variance explained for regression tasks]? For unsupervised/generative models this is a more tricky question - imputation or a supervised task using the 
    
3. Where will you get data? What sample size is available (if appropriate)? What format is it in? What preprocessing (if any) will you need to do? 
    
4. Reference at least 3 highly relevant/related papers (arXiv/bioRxiv is fine). If you’re using Latex you should learn how to use bibtex/biblatex. In Google docs I recommend using Paperpile. These should be referenced as you would see in a paper, e.g. 
    

Maurano, M. T. et al. Systematic localization of common disease-associated variation in regulatory DNA. Science 337, 1190–1195 (2012).

(adding a link so we can look them up easily would also be helpful). 

5. What are you planning to get done for the interim report, e.g. have downloaded, processed and understood the data, done some [EDA](https://en.wikipedia.org/wiki/Exploratory_data_analysis), maybe run a baseline method or code from a related method. 
    
6. How will you assess performance compared to existing/baseline approaches? 
    
7. Come up with a title: ideally this would give some sense of the problem and the type of method you plan to use (e.g. “Predicting single-cell chromatin accessibility from sequence using convolutional neural networks”). It’s ok if you don’t know what method you’ll use in the end. 
    

## Grading criteria

Clarity 

Did you effectively convey your research problem, the motivation, and your proposed approach? Be sure to make explicit what your contributions were not just in terms of figures and tables, but more importantly in terms of the ideas (biological and/or computational). I recommend checking your writing/grammar with something like Grammarly (we get free access through Columbia). 

Originality 

How original is the idea? Note that we don’t expect every project to solve a previously

unsolved problem. But we do expect projects to introduce some new computational idea to whatever problem they tackle.

Challenge (low grading weight for proposal)

How challenging was the project? 

- Developing a novel model from scratch > extending existing code >> applying off-the-shelf tools.
    
- Integrating multiple data types (e.g. RNA-seq and ATAC-seq) > just using one (e.g. predicting some outcome from RNA-seq like in Assignment 1). 
    

Relevance 

Is the problem relevant to the course? Hopefully yes after getting feedback on the pre-proposal!

**



**

### Predicting Unseen Multi-Gene Perturbation Responses with Attention-Based Aggregation

Name: Pramika Garg  
UNI: pg2899

Biological question:  
Can we better predict how a cell’s gene expression changes when two genes are perturbed together, especially for gene combinations that were not observed during training? The combined effect of two perturbations can be non-additive: activating genes A and B together may produce a response that cannot be predicted simply by adding their individual effects. I want to test whether explicitly modeling interactions between perturbations improves prediction of these unseen combinations.

2. ML methods/models:  
I will build on GEARS, a graph neural network that predicts post-perturbation gene expression from an unperturbed expression profile and a set of perturbed genes. GEARS represents each perturbation with a learned embedding and combines multiple perturbations using a sum operation followed by an MLP. My main extension will replace this fixed sum with a Set Transformer, which uses attention to learn how the effects of two perturbed genes interact while remaining independent of their ordering. I will compare the original GEARS sum aggregation, a simple mean aggregation baseline, and the attention-based model. The input will be the control-cell expression profile and identities of the perturbed genes; the output will be the predicted post-perturbation gene-expression vector. Performance on held-out gene pairs will be evaluated using mean squared error, Pearson correlation, and recovery of differentially expressed genes.

3. Data:  
I will use Norman et al. Perturb-seq dataset (GSE133344), which contains approximately 110,000 single cells across 287 perturbation conditions, including 132 gene pairs as well as the corresponding single-gene perturbations. GEARS already provides preprocessing and loading support for this dataset, which should keep the project focused on modeling rather than data cleaning.

4. Relevant paper:  
Roohani, Y., Huang, K. & Leskovec, J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. Nature Biotechnology 42, 927–935 (2024).

**

# Feedback

Good. The one other aspect you could consider is whether different gene embeddings could be added and improve performance. [https://www.biorxiv.org/content/10.1101/2025.06.27.661992v2](https://www.biorxiv.org/content/10.1101/2025.06.27.661992v2) has some interesting findings on that front.


