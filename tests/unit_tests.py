import unittest
import otscomics
import pandas as pd
import numpy as np
import anndata as ad

class TestOTscomics(unittest.TestCase):

    def setUp(self) -> None:
        """Initial loading of the data.
        """        

        # Load the data.
        # TODO: put directly in folder.
        data = pd.read_csv('data/liu_scrna_preprocessed.csv.gz', index_col=0)

        # Retrieve the clusters.
        clusters = np.array([col.split('_')[-1] for col in data.columns])

        # Select highly variable genes.
        data = data.iloc[np.argsort(data.std(1))[::-1][:1_000]]
        
        # Converting to AnnData for the rest of the analysis.
        self.adata = ad.AnnData(data.T)
        self.adata.obs['cell_line'] = clusters
    
    def test_data(self) -> None:
        """Test that the data was loaded properly.
        """

        # Test the dimensions of the data.
        self.assertEqual(self.adata.n_obs, 206)
        self.assertEqual(self.adata.n_vars, 1_000)

    def test_cost_matrix(self):
        """Test that the cost matrix is computed properly.
        """
        
        # Generate the cost matrix C.
        C = otscomics.cost_matrix(
            self.adata.X.T, cost='cosine', normalize_features=True)
        
        # Test the dimensions of C.
        self.assertEqual(C.shape, (1_000, 1_000))

        # Test that C has zero diagonal.
        np.testing.assert_almost_equal(np.diag(C), np.zeros(1_000))

        # Test that C has a max of 1.
        np.testing.assert_almost_equal(C.max(), 1)
        

if __name__ == '__main__':
    unittest.main()