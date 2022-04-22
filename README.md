[![DOI](https://zenodo.org/badge/344845096.svg)](https://zenodo.org/badge/latestdoi/344845096) [![Documentation Status](https://readthedocs.org/projects/ot-scomics/badge/?version=latest)](https://ot-scomics.readthedocs.io/en/latest/?badge=latest)

# Optimal Transport improves cell-cell similarity inference in single-cell omics data

This Python package will allow you to replicate the experiments from our research on applying Optimal Transport as a similarity metric in between single-cell omics data.

![image](https://user-images.githubusercontent.com/30904288/110963850-da6c0000-8352-11eb-8c0a-f725c1736169.png)

## Optimal Transport for single-cell omics

We propose the use of Optimal Transport (OT) as a cell-cell similarity metric for single-cell omics data. The code in this repository implements entropic-regularized OT distance computation with PyTorch, and applies it to public datasets of single-cell omics data. We compare the results to commonly used metrics like the euclidean distance or Pearson correlation, and demonstrate that OT increases performances in cell-cell similarity inference.

## Installing the package

    pip install otscomics

## Documentation

[https://ot-scomics.rtfd.io](https://ot-scomics.rtfd.io)

## Jupyter Notebook

The documentation includes a Jupyter Notebook demonstrating the package. While this notebook can be run un CPU, computations will be faster on GPU.

If you do not have access to a GPU you may want to use the (free) Google Colab platform to run this notebook: [colab.research.google.com/github/ComputationalSystemsBiology/.../OT_scOmics.ipynb](https://colab.research.google.com/github/ComputationalSystemsBiology/OT-scOmics/blob/main/docs/source/vignettes/OT_scOmics.ipynb).

Please make sure that the GPU is enabled. Navigate to "Edit→Notebook Settings" and select "GPU" from the "Hardware Accelerator" drop-down.

## Citing us

The paper describing OT-scOmics has been published on Bioinformatics.

[Open Access link](https://doi.org/10.1093/bioinformatics/btac084) (Huizing, Peyré, Cantini, 2022)

    @article{10.1093/bioinformatics/btac084,
		author = {Huizing, Geert-Jan and Peyré, Gabriel and Cantini, Laura},
		title = "{Optimal transport improves cell–cell similarity inference in single-cell omics data}",
		journal = {Bioinformatics},
		volume = {38},
		number = {8},
		pages = {2169-2177},
		year = {2022},
		month = {02},
		issn = {1367-4803},
		doi = {10.1093/bioinformatics/btac084},
		url = {https://doi.org/10.1093/bioinformatics/btac084},
		eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/8/2169/43370259/btac084.pdf},
	}
