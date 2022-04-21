import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import ot
from opt_einsum import contract as einsum

def cost_matrix(data, cost, normalize_features=True):
  """
  Compute cost matrix on input `data` for function `cost`
  Features are normalized by default (`normalize_features`)
  """
  if normalize_features:
    sums = data.sum(1).reshape(-1, 1)
    C = cdist(data/sums, data/sums, metric=cost)
  else:
    C = cdist(data, data, metric=cost)
  return C/C.max()

def OT_distance_matrix(
  data_np, cost, eps=.1, max_iter=100,
  n_batches=10, threshold=1e-5,
  dtype=torch.float32, device='cuda',
  divide_max=True, numItermax=500,
  stopThr=1e-5, batch_size=200):
  """
  Compute OT distance matrix.
  `data_np`: data as a NumPy matrix (on CPU)
  `cost`: cost matrix as a NumPy array (on CPU)
  `eps`: regularization parameter
  `max_iter`: maximum number of iterations of the Sinkhorn algorithm
  `n_batches`: how many batches for the computation (more batches, less memory)
  `dtype`: should be `torch.float32` or `torch.double`
  `threshold`: precision of the computation
  `device`: `cuda` or `cpu`
  `divide_max`: Divides the distance matrix by its maximum before returning.
  For small values of `epsilon`, `max_iter` should be bigger to converge better
  In order to use OT in your own application, you may want to consider the more
  flexible and canonical tools POT (https://pythonot.github.io/),
  OTT (https://ott-jax.readthedocs.io/) or Geomloss (https://www.kernel-operations.io/geomloss/).
  """

  # Move the cost to PyTorch.
  C = torch.from_numpy(cost)
  C = C.to(device=device, dtype=dtype)

  # Compute the kernel
  K = torch.exp(-C/eps)

  data = torch.from_numpy(data_np)
  data = data.to(device=device, dtype=dtype)

  # Define batches
  idx_i, idx_j = np.tril_indices(data.shape[1])
  idx_i = np.array_split(idx_i, n_batches)
  idx_j = np.array_split(idx_j, n_batches)

  D = torch.zeros(data.shape[1], data.shape[1], device='cpu', dtype=dtype)

  pbar = tqdm(total=data.shape[1]*(data.shape[1] - 1)//2, leave=False)

  errors = []

  # Iterate over the lines.
  for i in range(data.shape[1]):
    for ii in np.split(range(i+1), np.arange(batch_size, i+1, batch_size)):

      # Compute the Sinkhorn dual variables
      _, wass_log = ot.sinkhorn(
          data[:,i].contiguous(), # This is the source histogram.
          data[:,ii].contiguous(), # These are the target histograms.
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
          f*data[:,[i]*len(ii)] +
          g*data[:,ii] -
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

def C_index(D, clusters):
  """
  Compute C index given:
  - a distance matrix `D`
  - cluster assignments `clusters`
  
  Value between 0 (best) and 1 (worst)
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