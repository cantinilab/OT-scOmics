# Optimal Transport improves cell-cell similarity inference in single-cell omics data

This Jupyter Notebook will walk you trough the code to replicate the experiments from our research on applying Optimal Transport as a similarity metric in between single-cell omics data.

![image](https://user-images.githubusercontent.com/30904288/110963850-da6c0000-8352-11eb-8c0a-f725c1736169.png)

## Optimal Transport for single-cell omics

We propose the use of Optimal Transport (OT) as a cell-cell similarity metric for single-cell omics data. The code in this repository implements entropic-regularized OT distance computation with PyTorch, and applies it to public datasets of single-cell omics data. We compare the results to commonly used metrics like the euclidean distance or Pearson correlation, and demonstrate that OT increases performances in cell-cell similarity inference.

## Running on Colab GPU

This notebook is designed to be run on GPU. If you do not have access to a GPU you may want to use the (free) Google Colab platform to run this notebook. For that, you can simply access the notebook at this link: [colab.research.google.com/github/ComputationalSystemsBiology/.../OT_scOmics.ipynb](https://colab.research.google.com/github/ComputationalSystemsBiology/OT-scOmics/blob/main/OT_scOmics.ipynb).

Please note that you will have to import the preprocessed csv data to the Colab runtime.

Please make sure that the GPU is enabled. Navigate to "Edit→Notebook Settings" and select "GPU" from the "Hardware Accelerator" drop-down.

## Citing us

The preprint describing OT-scOmics is available in BioRxiv 

[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2021.03.19.436159v1) (Huizing, Peyré, Cantini, 2021)

    @article {Huizing2021.03.19.436159,
    	author = {Huizing, Geert-Jan and Peyre, Gabriel and Cantini, Laura},
	    title = {Optimal Transport improves cell-cell similarity inference in single-cell omics data},
    	elocation-id = {2021.03.19.436159},
    	year = {2021},
    	doi = {10.1101/2021.03.19.436159},
    	publisher = {Cold Spring Harbor Laboratory},
    	abstract = {The recent advent of high-throughput single-cell molecular profiling is revolutionizing biology and medicine by unveiling the diversity of cell types and states contributing to development and disease. The identification and characterization of cellular heterogeneity is typically achieved through unsupervised clustering, which crucially relies on a similarity metric. We here propose the use of Optimal Transport (OT) as a cell-cell similarity metric for single-cell omics data. OT defines distances to compare, in a geometrically faithful way, high-dimensional data represented as probability distributions. It is thus expected to better capture complex relationships between features and produce a performance improvement over state-of-the-art metrics. To speed up computations and cope with the high-dimensionality of single-cell data, we consider the entropic regularization of the classical OT distance. We then extensively benchmark OT against state-of-the-art metrics over thirteen independent datasets, including simulated, scRNA-seq, scATAC-seq and single-cell DNA methylation data. First, we test the ability of the metrics to detect the similarity between cells belonging to the same groups (e.g. cell types, cell lines of origin). Then, we apply unsupervised clustering and test the quality of the resulting clusters. In our in-depth evaluation, OT is found to improve cell-cell similarity inference and cell clustering in all simulated and real scRNA-seq data, while its performances are comparable with Pearson correlation in scATAC-seq and single-cell DNA methylation data. All our analyses are reproducible through the OT-scOmics Jupyter notebook available at https://github.com/ComputationalSystemsBiology/OT-scOmics.Competing Interest StatementThe authors have declared no competing interest.},
    	URL = {https://www.biorxiv.org/content/early/2021/03/20/2021.03.19.436159},
    	eprint = {https://www.biorxiv.org/content/early/2021/03/20/2021.03.19.436159.full.pdf},
    	journal = {bioRxiv}
    }
