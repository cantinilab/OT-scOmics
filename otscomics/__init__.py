import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
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

def OT_distance_matrix(data_np, cost, eps=.1, max_iter=100, n_batches=10, threshold=1e-5, dtype=torch.float32, device='cuda', divide_max=True):
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

  # Compute K (inplace)
  K = torch.from_numpy(cost)
  K = K.to(device=device, dtype=dtype)
  
  K.mul_(-1/eps)
  K.exp_()

  data = torch.from_numpy(data_np)
  data = data.to(device=device, dtype=dtype)

  # Define batches
  idx_i, idx_j = np.tril_indices(data.shape[1])
  idx_i = np.array_split(idx_i, n_batches)
  idx_j = np.array_split(idx_j, n_batches)

  D = torch.zeros(data.shape[1], data.shape[1], device='cpu', dtype=dtype)

  errors = []

  for k in tqdm(range(n_batches)):
    u = torch.ones(data[:,idx_i[k]].shape, device=device, dtype=dtype)
    v = torch.ones(data[:,idx_j[k]].shape, device=device, dtype=dtype)

    err_u, err_v = [], []
    i = 0
    d = 0
    while len(err_u) == 0 or (max(err_u[-1], err_v[-1]) > threshold and i < max_iter):
      i += 1

      torch.matmul(K, v, out=u)
      u.pow_(-1).mul_(data[:,idx_i[k]])

      torch.matmul(K, u, out=v)
      v.pow_(-1).mul_(data[:,idx_j[k]])
      
      if i % 5 == 0:
        err_u.append(torch.norm(einsum('ik,ij,jk->ik', u, K, v) - data[:,idx_i[k]], p=1).cpu()/len(idx_i[k]))
        err_v.append(torch.norm(einsum('ik,ij,jk->jk', u, K, v) - data[:,idx_j[k]], p=1).cpu()/len(idx_j[k]))
    
    errors.append((err_u, err_v))

    D[idx_i[k], idx_j[k]] = eps*((torch.log(u)*data[:,idx_i[k]]).sum(0) + (torch.log(v)*data[:,idx_j[k]]).sum(0) - ((K @ v) * u).sum(0)).cpu()

  # Unbias distance matrix
  d = torch.diag(D)
  d = d + d.reshape(-1, 1)
  D = D + D.T - .5*d
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