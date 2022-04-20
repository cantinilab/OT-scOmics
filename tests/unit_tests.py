import unittest
import otscomics
import pandas as pd
import numpy as np
import anndata as ad

class TestOTscomics(unittest.TestCase):

    def setUp(self) -> None:

        # Load the data.
        # TODO: put directly in folder.
        data = pd.read_csv('data/liu_scrna_preprocessed.csv.gz', index_col=0)

        # Retrieve the clusters.
        clusters = np.array([col.split('_')[-1] for col in data.columns])
        idx = np.argsort(clusters) # Sorted indices (for visulization)

        # Select highly variable genes.
        data = data.iloc[np.argsort(data.std(1))[::-1][:1_000]]
        
        # Converting to AnnData for the rest of the analysis.
        self.adata = ad.AnnData(data.T)
        self.adata.obs['cell_line'] = clusters


    def test_nothing(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()