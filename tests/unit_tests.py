from re import A
import unittest
import otscomics
import pandas as pd
import numpy as np
import anndata as ad
import torch
import ot
from scipy.spatial.distance import cdist

def ot_distance(a, b, C, eps):
    cost = torch.from_numpy(C)
    K = torch.exp(-cost/eps)

    _, wass_log = ot.sinkhorn(
        a.contiguous(), b.contiguous(),
        cost, reg=eps, log=True)

    # Compute the exponential dual potentials.
    f, g = eps*wass_log['u'].log(), eps*wass_log['v'].log()

    # Compute the Sinkhorn cost.
    return torch.sum(f*a + g*b - eps*wass_log['u']*(K@wass_log['v']))

def sinkhorn_divergence(a, b, C, eps):
    ab = ot_distance(a, b, C, eps)
    aa = ot_distance(a, a, C, eps)
    bb = ot_distance(b, b, C, eps)
    return ab - .5*(aa + bb)

class TestOTscomics(unittest.TestCase):

    def setUp(self) -> None:
        """Initial loading of the data.
        """

        # Load the data.
        data = pd.read_csv('data/liu_scrna_preprocessed.csv.gz', index_col=0)

        # Retrieve the clusters.
        clusters = np.array([col.split('_')[-1] for col in data.columns])

        # Select highly variable genes.
        self.n_obs, self.n_vars = 206, 500
        data = data.iloc[np.argsort(data.std(1))[::-1][:self.n_vars]]
        
        # Converting to AnnData for the rest of the analysis.
        self.adata = ad.AnnData(data.T)
        self.adata.obs['cell_line'] = clusters
    
    def test_data(self) -> None:
        """Test that the data was loaded properly.
        """

        # Test the dimensions of the data.
        self.assertEqual(self.adata.n_obs, self.n_obs)
        self.assertEqual(self.adata.n_vars, self.n_vars)

    def test_cost_matrix(self) -> None:
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
    
    def test_ot_distance(self) -> None:
        """Test the pairwise OT distance.
        """

        eps = .5

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
        D_ot, _ = otscomics.OT_distance_matrix(
            data_norm, cost=C, eps=eps,
            dtype=torch.double, device='cpu',
            divide_max=False)

        # Test that D has the right shape.
        self.assertEqual(D_ot.shape, (self.n_obs, self.n_obs))
        
        # Check that D is positive.
        self.assertEqual(np.sum(D_ot < 0), 0)

        # Test that D is symmetric.
        np.testing.assert_almost_equal(D_ot, D_ot.T)

        # Test that D has zero diagonal.
        np.testing.assert_almost_equal(np.diag(D_ot), np.zeros(self.n_obs))

        # Test that D contains the right distances.
        i, j = np.random.choice(np.arange(self.n_obs), size=2, replace=False)
        a = torch.from_numpy(data_norm[:,i])
        b = torch.from_numpy(data_norm[:,j])
        np.testing.assert_almost_equal(
            D_ot[i, j],
            sinkhorn_divergence(a, b, C, eps))
        
    
    def test_c_index(self) -> None:
        """Test the simple C index function
        """

        # Create an Euclidean distance matrix.
        D = cdist(self.adata.X, self.adata.X)

        # Compute the C index.
        score = otscomics.C_index(D, self.adata.obs['cell_line'])

        # Check that the score is between 0 and 1.
        self.assertGreaterEqual(score, 0)
        self.assertGreaterEqual(1, score)
        

if __name__ == '__main__':
    unittest.main()