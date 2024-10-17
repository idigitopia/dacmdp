from collections import defaultdict
import torch
import warnings
# from knn_cuda import KNN as KNN_CUDA
import math


import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor



class THelper():

    @staticmethod
    def lookup_index_numerically(vector:torch.Tensor, matrix:torch.Tensor)->int:
        matches = (torch.sum(matrix == vector, dim=1) == len(vector)).nonzero()
        if len(matches) > 1:
            assert False, matches
        elif len(matches) == 1:
            return matches.item()
        else:
            warnings.warn('no index found, returning -1')
            return -1

    @staticmethod
    def lookup_index_by_hash(vector: torch.Tensor, torch_matrix:torch.Tensor)->int:
        """
        Returns a function for quick lookup for an index of a tensor.
        Only Factors in for 5 Decimal point

        Returns:
            _type_: _description_
        """
        index_by_hash = defaultdict(None, {hash(r): i for i, r in enumerate(torch_matrix)})
        return index_by_hash[hash(vector)]
    
    @staticmethod
    def lookup_nn_index(vector: torch.Tensor, torch_matrix:torch.Tensor)->int:
        """
        Returns a function for quick lookup for an index of a tensor.
        Only Factors in for 5 Decimal point

        Returns:
            _type_: _description_
        """
        nn_index = THelper.calc_knn_indices(vector, torch_matrix, 1)[0]
        return nn_index

    @staticmethod
    def calc_knn_indices(query: torch.Tensor, data: torch.Tensor, k):
        dist = torch.norm(data - query, dim=1, p=2)
        knn = dist.topk(k, largest=False)
        return knn.indices

    
    @staticmethod
    @torch.jit.script
    def calc_knn_indices_jit(query: torch.Tensor, data: torch.Tensor, k:int):
        dist = torch.norm(data - query, dim=1, p=2)
        knn = dist.topk(k, largest=False)
        return knn.indices

    @staticmethod
    def calc_indices_between(query: torch.Tensor, data: torch.Tensor, dist_min, dist_max, k):
        dist = torch.norm(data - query, dim=1, p=2)
        knn = dist.topk(k, largest=False)
        return knn.indices

    @staticmethod
    def calc_knn(query: torch.Tensor, data: torch.Tensor, k:int):
        dist = torch.norm(data - query, dim=1, p=2)
        knn = dist.topk(k, largest=False)
        return knn

    @staticmethod
    def batch_calc_knn(query_batch: torch.Tensor, data: torch.Tensor, k:int):
        dists = torch.norm(query_batch.unsqueeze(1) - data.unsqueeze(0), dim=2, p=2)
        nn_dists, nn_idx = torch.topk(dists, k, dim=-1, largest=False)
        return nn_idx, nn_dists
    
    @staticmethod
    def batch_calc_knn_ret_flat(query_batch: torch.Tensor, data: torch.Tensor, k:int):
        nn_idx, nn_dists = THelper.batch_calc_knn(query_batch, data, k)
        return nn_idx.view(-1), nn_dists.view(-1)

    @staticmethod
    @torch.jit.script
    def calc_knn_jit(query: torch.Tensor, data: torch.Tensor, k:int):
        dist = torch.norm(data - query, dim=1, p=2)
        knn = dist.topk(k, largest=False)
        return knn
    
    @staticmethod
    @torch.jit.script
    def batch_calc_knn_jit(query_batch: torch.Tensor, data: torch.Tensor, k:int):
        dists = torch.norm(query_batch.unsqueeze(1) - data.unsqueeze(0), dim=2, p=2)
        nn_dists, nn_idx = torch.topk(dists, k, dim=-1, largest=False)
        return nn_idx, nn_dists
    
    @staticmethod
    @torch.jit.script
    def batch_calc_knn_ret_flat_jit(query_batch: torch.Tensor, data: torch.Tensor, k:int):
        nn_idx, nn_dists = THelper.batch_calc_knn(query_batch, data, k)
        return nn_idx.view(-1), nn_dists.view(-1)
    
    @staticmethod
    def batch_calc_knn_pykeops(query: torch.Tensor, data: torch.Tensor, k:int):
        X_i = LazyTensor(query[:, None, :])  # (10000, 1, 784) test set
        X_j = LazyTensor(data[None, :, :])  # (1, 60000, 784) train set

        D_ij = ((X_i - X_j) ** 2).sum(-1)  # (10000, 60000) symbolic matrix of squared L2 distances
        ind_knn = D_ij.argKmin(k, dim=1)  # Samples <-> Dataset, (N_test, K)
        dist_knn = torch.norm(query.unsqueeze(1) - data[ind_knn], p = 2, dim = -1)
        return ind_knn, dist_knn

    @staticmethod
    def batch_calc_knn_ret_flat_pykeops(query: torch.Tensor, data: torch.Tensor, k: int):
        nn_idx, nn_dists = THelper.batch_calc_knn_pykeops(query, data, k)
        return nn_idx.view(-1), nn_dists.view(-1)



def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64
    device_id = "cuda:0" if use_cuda else "cpu"
    
    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""


    use_cuda = torch.cuda.is_available()
    dtype = torch.float32 if use_cuda else torch.float64
    device_id = "cuda:0" if use_cuda else "cpu"

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c