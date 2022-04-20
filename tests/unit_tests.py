import unittest
import otscomics
import pandas as pd
import numpy as np
import anndata as ad
import torch

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
        self.n_vars = 500
        data = data.iloc[np.argsort(data.std(1))[::-1][:self.n_vars]]
        
        # Converting to AnnData for the rest of the analysis.
        self.adata = ad.AnnData(data.T)
        self.adata.obs['cell_line'] = clusters
    
    def test_data(self) -> None:
        """Test that the data was loaded properly.
        """

        # Test the dimensions of the data.
        self.assertEqual(self.adata.n_obs, 206)
        self.assertEqual(self.adata.n_vars, self.n_vars)

    def test_cost_matrix(self):
        """Test that the cost matrix is computed properly.
        """
        
        # Generate the cost matrix C.
        C = otscomics.cost_matrix(
            self.adata.X.T.astype(np.double),
            cost='cosine', normalize_features=True)
        
        # Test the dimensions of C.
        self.assertEqual(C.shape, (self.n_vars, self.n_vars))

        # Test that C has zero diagonal.
        np.testing.assert_almost_equal(np.diag(C), np.zeros(self.n_vars))

        # Test that C is symmetric.
        np.testing.assert_almost_equal(C, C.T)

        # Test that C has a max of 1.
        np.testing.assert_almost_equal(C.max(), 1)

        # Check positivity.
        self.assertEqual(np.sum(C < 0), 0)
    
    def test_ot_distance(self):
        """Test the pairwise OT distance.
        """

        # Per-cell normalization (mandatory).
        data_norm = self.adata.X.T.astype(np.double)
        data_norm /= data_norm.sum(0)

        # Add a small value to avoid numerical errors.
        data_norm += 1e-9
        data_norm /= data_norm.sum(0)
        
        # Compute the ground cost.
        C = otscomics.cost_matrix(
            self.adata.X.T.astype(np.double),
            cost='cosine', normalize_features=True)
        
        # Compute the OT distance matrix.
        D_ot, errors = otscomics.OT_distance_matrix(
            data_norm, cost=C, eps=.5,
            n_batches=10, threshold=1e-8,
            max_iter=100, dtype=torch.double, device='cpu')
        
        # Check positivity
        self.assertEqual(np.sum(D_ot < 0), 0)
        

if __name__ == '__main__':
    unittest.main()