# Optimal Transport improves cell-cell similarity inference in single-cell omics data

This Jupyter Notebook will walk you trough the code to replicate the experiments from our research on applying Optimal Transport as a similarity metric in between single-cell omics data.

![image](https://user-images.githubusercontent.com/30904288/110963850-da6c0000-8352-11eb-8c0a-f725c1736169.png)

## Optimal Transport for single-cell omics

We propose the use of Optimal Transport (OT) as a cell-cell similarity metric for single-cell omics data. The code in this repository implements entropic-regularized OT distance computation with PyTorch, and applies it to public datasets of single-cell omics data. We compare the results to commonly used metrics like the euclidean distance or Pearson correlation, and demonstrate that OT increases performances in cell-cell similarity inference.

## Running on Colab GPU

While this notebook can be run un CPU, computations will be faster on GPU. If you do not have access to a GPU you may want to use the (free) Google Colab platform to run this notebook. For that, you can simply access the notebook at this link: [colab.research.google.com/github/ComputationalSystemsBiology/.../OT_scOmics.ipynb](https://colab.research.google.com/github/ComputationalSystemsBiology/OT-scOmics/blob/main/OT_scOmics.ipynb).

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
    	URL = {https://www.biorxiv.org/content/early/2021/03/20/2021.03.19.436159},
    	eprint = {https://www.biorxiv.org/content/early/2021/03/20/2021.03.19.436159.full.pdf},
    	journal = {bioRxiv}
    }
