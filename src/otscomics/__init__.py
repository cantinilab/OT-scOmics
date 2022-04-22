from typing import Iterable
import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import ot

def cost_matrix(
  data: np.ndarray, cost: str = 'correlation',
  normalize_features: bool = True) -> np.ndarray:
  """Compute an empirical ground cost matrix, i.e. a pairwise distance matrix
  between the rows of the dataset (l1-normalized by default). Accepted
  distances are the ones compatible with Scipy's `cdist`.

  Args:
      data (np.ndarray):
        The input data, samples as columns and features as rows.
      cost (str):
        The metric use. Defaults to `correlation`.
      normalize_features (bool, optional):
        Whether to divide the rows by their sum before
        computing distances. Defaults to True.

  Returns:
      np.ndarray: The pairwise cost matrix.
  """  
  if normalize_features:
    sums = data.sum(1).reshape(-1, 1)
    C = cdist(data/sums, data/sums, metric=cost)
  else:
    C = cdist(data, data, metric=cost)
  return C/C.max()

def OT_distance_matrix(
  data: np.ndarray,
  cost: np.ndarray,
  eps: float = .1,
  dtype: torch.dtype = torch.double,
  device: str = 'cuda',
  divide_max: bool = False,
  numItermax: int = 500,
  stopThr: float = 1e-5,
  batch_size: int = 200) -> np.ndarray:
  """Compute the pairwise Optimal Transport distance matrix. We compute
  Sinkhorn Divergences using POT's implementation of the Sinkhorn algorithm.
  Computations are done using PyTorch on a specified device. But the result is
  a numpy array. This allows not saturating the GPU for large matrices.

  Args:
      data (np.ndarray):
        The input data, as a numpy array.
      cost (np.ndarray):
        The ground cost between features.
      eps (float, optional):
        The entropic regularization parameter. Small regularization requires
        more iterations and double precision. Defaults to .1.
      dtype (torch.dtype, optional):
        The torch dtype used for computations. Double is more precise but
        takes up more space. Defaults to torch.double.
      device (str, optional):
        The torch device to compute on, typically 'cpu' or 'cuda'.
        Defaults to 'cuda'.
      divide_max (bool, optional):
        Whether to divide the resulting matrix by its maximum value.
        This can be useful to compare matrices. Defaults to False.
      numItermax (int, optional):
        Used by POT, maximum number of Sinkhorn iterations. Defaults to 500.
      stopThr (float, optional):
        Used by POT, tolerance for early stopping in the Sinkhorn iterations.
        Defaults to 1e-5.
      batch_size (int, optional):
        The batch size, i.e. how many distances can be computed at the same
        time. Should be as large as possible on your hardware. Defaults to 200.

  Returns:
      np.ndarray: The pairwise OT distance matrix.
  """  

  # Move the cost to PyTorch.
  C = torch.from_numpy(cost)
  C = C.to(device=device, dtype=dtype)

  # Compute the kernel
  K = torch.exp(-C/eps)

  data_tensor = torch.from_numpy(data)
  data_tensor = data_tensor.to(device=device, dtype=dtype)

  D = torch.zeros(data_tensor.shape[1], data_tensor.shape[1], device='cpu', dtype=dtype)

  pbar = tqdm(total=data_tensor.shape[1]*(data_tensor.shape[1] - 1)//2, leave=False)

  errors = []

  # Iterate over the lines.
  for i in range(data_tensor.shape[1]):
    for ii in np.split(range(i+1), np.arange(batch_size, i+1, batch_size)):

      # Compute the Sinkhorn dual variables
      _, wass_log = ot.sinkhorn(
          data_tensor[:,i].contiguous(), # This is the source histogram.
          data_tensor[:,ii].contiguous(), # These are the target histograms.
          C, # This is the ground cost.
          eps, # This is the regularization parameter.
          log=True, # Return the dual variables
          stopThr=stopThr,
          numItermax=numItermax
      )

      # Compute the exponential dual potentials.
      f, g = eps*wass_log['u'].log(), eps*wass_log['v'].log()

      if len(wass_log['err']) > 0:
        errors.append(wass_log['err'][-1])

      # Compute the Sinkhorn costs.
      # These will be used to compute the Sinkhorn divergences
      wass = (
          f*data_tensor[:,[i]*len(ii)] +
          g*data_tensor[:,ii] -
          eps*wass_log['u']*(K@wass_log['v'])
      ).sum(0)

      # Add them in the distance matrix (including symmetric values).
      D[i,ii] = D[ii,i] = wass.cpu()

      pbar.update(len(ii))
    
  pbar.close()
  
  # Get the diagonal terms OT_eps(a, a).
  d = torch.diagonal(D)

  # The Sinkhorn divergence is OT(a, b) - (OT(a, a) + OT(b, b))/2.
  D = D - .5*(d.view(-1, 1) + d.view(1, -1))

  # Make sure there are no negative values.
  assert((D < 0).sum() == 0)

  # Make sure the diagonal is zero.
  D.fill_diagonal_(0)

  if divide_max:
    D /= torch.max(D)

  return D.numpy(), errors

def C_index(D: np.ndarray, clusters: np.ndarray) -> float:
  """Compute the C index, a measure of how well the pairwise distances reflect
  ground truth clusters. Implemented here for reference, but the silhouette
  score (aka Average Silhouette Width) is a more standard metric for this.

  Args:
      D (np.ndarray): The pairwise distances.
      clusters (np.ndarray): The ground truth clusters.

  Returns:
      float: The C index.
  """
  Sw = Nw = 0
  for c in np.unique(clusters):
    idx = np.where(clusters == c)[0]
    Sw += D[idx][:,idx].sum()/2
    Nw += int(len(idx)*(len(idx) - 1)/2)

  els = []
  for i in range(len(D)):
    for j in range(i):
      els.append(D[i, j])
  Smin = np.sort(np.array(els))[:Nw].sum()
  Smax = np.sort(np.array(els))[::-1][:Nw].sum()

  return (Sw - Smin)/(Smax - Smin)