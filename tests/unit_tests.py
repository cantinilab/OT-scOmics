import unittest
import otscomics
import pandas as pd
import numpy as np

class TestOTscomics(unittest.TestCase):

    def setUp(self) -> None:

        # Load the data.
        # TODO: put directly in folder.
        data_path = 'data/liu_scrna_preprocessed.csv.gz'
        self.data = pd.read_csv(data_path, index_col=0)

        # Retrieve the clusters.
        clusters = np.array([col.split('_')[-1] for col in self.data.columns])
        idx = np.argsort(clusters) # Sorted indices (for visulization)

        # Select highly variable genes.
        self.data = self.data.iloc[np.argsort(self.data.std(1))[::-1][:1_000]]


    def test_nothing(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()