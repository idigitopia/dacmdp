from collections import defaultdict
import torch
import warnings
# from knn_cuda import KNN as KNN_CUDA
import math
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
    def batch_calc_knn_ret_flat(query_batch: torch.Tensor, data: torch.Tensor, k:int):
        knn_batch = [THelper.calc_knn(q, data,k) for q in query_batch]
        knn_indices_flat = torch.concat([knn.indices for knn in knn_batch])
        knn_values_flat = torch.concat([knn.values for knn in knn_batch])
        return knn_indices_flat, knn_values_flat


    
    @staticmethod
    @torch.jit.script
    def batch_calc_knn_jit(query_batch: torch.Tensor, data: torch.Tensor, k:int):
        dists = torch.cdist(query_batch, data)
        nn_dists, nn_idx = torch.topk(dists, k, dim=-1, largest = False)
        return nn_idx, nn_dists
    
    @staticmethod
    @torch.jit.script
    def batch_calc_knn_ret_flat_jit(query_batch: torch.Tensor, data: torch.Tensor, k:int):
        dists = torch.cdist(query_batch, data)
        nn_dists, nn_idx = torch.topk(dists, k, dim=-1, largest = False)
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

    # @staticmethod
    # def batch_calc_knn_cuda(query_batch: torch.Tensor, data: torch.Tensor, k:int):
    #     knn_kernel = KNN_CUDA(k, transpose_mode=True)
    #     nn_dists, nn_idx = knn_kernel(data, query_batch)
    #     return nn_idx, nn_dists
    
    # @staticmethod
    # def batch_calc_knn_ret_flat_cuda(query_batch: torch.Tensor, data: torch.Tensor, k: int):
    #     nn_idx, nn_dists = THelper.batch_calc_knn_cuda(query_batch, data, k)
    #     return nn_idx.view(-1), nn_dists.view(-1)
