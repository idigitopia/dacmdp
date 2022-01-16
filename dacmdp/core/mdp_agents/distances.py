# Default Python Pacvkages
import time
from collections import defaultdict

# Standard Python Packages.
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree as RawKDTree

# Project Specific Dependencies
from lmdp.data.buffer import iter_batch

def v_iter(iterator, verbose, message=""):
    """
    Returns a verbose iterator i.e. tqdm enabled iterator if verbose is True. 
    It also attaches the passed message to the iterator.
    """
    if verbose:
        vb_iterator = tqdm(iterator)
        vb_iterator.set_description(message)
    else:
        vb_iterator = iterator

    return vb_iterator

class MyKDTree():
    """
    Class to contain all the KD Tree related logics. 
    - Builds the index and inverseIndex for the vectors passed as the vocabulary for knn 
    - can get 1/k NN or 1/k NN of a batch of passed query vectors. 
    """

    def __init__(self, all_vectors):
        self.s2i, self.i2s = self._gen_vocab(all_vectors)
        self.KDtree = RawKDTree(np.array(list(self.s2i.keys())))

        self.get_knn = lambda s, k: self.get_knn_batch(np.array([s]), k)[0]
        self.get_nn = lambda s: list(self.get_knn_batch(np.array([s]), 1)[0])[0]
        self.get_nn_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_batch(s_batch, 1)]
        self.get_nn_sub_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_sub_batch(s_batch, 1)]
        self.get_knn_idxs = lambda s, k: self.get_knn_idxs_batch(np.array([s]), k)[0]
        self.get_nn_idx = lambda s: list(self.get_knn_idxs_batch(np.array([s]), 1)[0])[0]
        self.get_nn_idx_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_idxs_batch(s_batch, 1)]
        self.get_nn_idx_sub_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_idxs_sub_batch(s_batch, 1)]

    def _gen_vocab(self, all_vectors):
        """
        generate index mappings and inverse index mappings. 
        """

        s2i = {tuple(s): i for i, s in enumerate(all_vectors)}
        i2s = {i: tuple(s) for i, s in enumerate(all_vectors)}
        return s2i, i2s

    def get_knn_batch(self, s_batch, k):
        """
        input: a list of query vectors. 
        output: a list of k-nn tuples for each query vector. 
        """

        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        def get_nn_dict(dists, idxs): return {
            self.i2s[int(idx)]: dist for dist, idx in zip(dists, idxs)}
        nn_dict_list = [get_nn_dict(dists, idxs)
                        for dists, idxs in zip(dists_b, idxs_b)]
        return nn_dict_list

    def get_knn_idxs_batch(self, s_batch, k):
        """
        input: a list of query vectors. 
        output: a list of k-nn idxs for each query vector.
        """

        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        def get_nn_dict(dists, idxs): return {
            idx: dist for dist, idx in zip(dists, idxs)}
        nn_dict_list = [get_nn_dict(dists, idxs)
                        for dists, idxs in zip(dists_b, idxs_b)]
        return nn_dict_list

    # Get knn with smaller batch sizes. | useful when passing large batches.
    def get_knn_sub_batch(self, s_batch, k, batch_size=256, verbose=True, message=None):
        """
        # Get knn with smaller batch sizes. | useful when passing large batches.
        input: a large list of query vectors. 
        output: a large list of k-nn tuples for each query vector. 
        """

        nn_dict_list = []
        for small_batch in v_iter(iter_batch(s_batch, batch_size), verbose, message or "getting NN"):
            nn_dict_list.extend(self.get_knn_batch(small_batch, k))
        return nn_dict_list

    def get_knn_idxs_sub_batch(self, s_batch, k, batch_size=256, verbose=True, message=None):
        nn_dict_list = []
        for small_batch in v_iter(iter_batch(s_batch, batch_size), verbose, message or "getting NN Idxs"):
            nn_dict_list.extend(self.get_knn_idxs_batch(small_batch, k))
        return nn_dict_list

    @staticmethod
    def calc_prob_distr(knn_dist_dict, delta, force_uniform_distr=False):
        """
        Return a dictionary with normalized distances. can now be treated as probabilities.
        Input norm_by_dist: returns a uniform distribution if False, if True Normalize. 
        """
        # todo Add a choice to do exponential averaging here.
        if force_uniform_distr:
            all_knn_probs = {s: 1/len(knn_dist_dict)
                             for s, d in knn_dist_dict.items()}
        else:
            # get similarity scores
            all_knn_sim_scores = {nn: 1 / (dist + delta)
                                  for nn, dist in knn_dist_dict.items()}
            # convert similarity scores into probability distribution
            all_knn_probs = {nn: knn_kernel / sum(all_knn_sim_scores.values()) for nn, knn_kernel in
                             all_knn_sim_scores.items()}

        return all_knn_probs

# KD Tree helper function


class MyFactoredKDTree():
    def __init__(self, state_factor_pairs):
        """
        Emulates all apis of MyKDTree. but pretends that any state with different factors are infinitely apart. 
        """

        self.sf2i, self.i2sf = self._gen_vocab(state_factor_pairs)

        self.factored_states = defaultdict(lambda: [])  # factored state list
        for s, a in state_factor_pairs:
            self.factored_states[tuple(a)].append(s)

        self.factored_KDTrees = {tuple(a): MyKDTree(states)
                                 for a, states in self.action_factored_states.items()
                                 if len(states) > 1}
        # ToDo, does not handle the case where there is not even a single state for a given factor.

        self.v_print("kDTree built:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))

    def _gen_vocab(self, state_factor_pairs):
        """
        generate index mappings and inverse index mappings. 
        """
        sf2i = {tuple(sf_pair): i for i,
               sf_pair in enumerate(state_factor_pairs)}
        i2sf = {i: tuple(sf_pair)
               for i, sf_pair in enumerate(state_factor_pairs)}
        return sf2i, i2sf

    def get_knn_batch(self, state_factor_pairs, k, is_shared_factor=False):
        """
        input: a list of query state factor pairs. [(s1,a1), (s2, a2) . . ]
        output: a list of k-nn tuples for each query vector. 
        """
        s_batch = list(map(tuple, zip(*state_factor_pairs)[0]))
        f_batch = list(map(tuple, zip(*state_factor_pairs)[1]))

        if is_shared_factor:
            factor = f_batch[0]  # fetch shared factor
            s_nn_dict_list = self.factored_KDTrees[factor].get_knn_batch(
                s_batch, k)
        else:
            s_nn_dict_list = [self.factored_KDTrees[factor].get_knn(state, k)
                              for state, factor in zip(s_batch, f_batch)]

        sf_nn_dict_list = [{(k, factor): v for k, v in s_nn_dict.items()}
                           for s_nn_dict, factor in zip(s_nn_dict_list, f_batch)]
        return sf_nn_dict_list

    def get_knn_idxs_batch(self, state_factor_pairs, k, is_shared_factor=False):
        """
        input: a list of query vectors. 
        output: a list of k-nn idxs for each query vector.
        """
        s_batch = list(map(tuple, zip(*state_factor_pairs)[0]))
        f_batch = list(map(tuple, zip(*state_factor_pairs)[1]))

        if is_shared_factor:
            factor = f_batch[0]  # fetch shared factor
            s_nn_dict_list = self.factored_KDTrees[factor].get_knn_batch(
                s_batch, k)
        else:
            s_nn_dict_list = [self.factored_KDTrees[factor].get_knn(state, k)
                              for state, factor in zip(s_batch, f_batch)]

        sf_nnidx_dict_list = [{self.sf2i(tuple(k, factor)): v for k, v in s_nn_dict.items()}
                           for s_nn_dict, factor in zip(s_nn_dict_list, f_batch)]
        return sf_nnidx_dict_list

    # Get knn with smaller batch sizes. | useful when passing large batches.
    def get_knn_sub_batch(self, s_batch, k, batch_size=256, verbose=True, message=None):
        """
        # Get knn with smaller batch sizes. | useful when passing large batches.
        input: a large list of query vectors. 
        output: a large list of k-nn tuples for each query vector. 
        """

        nn_dict_list = []
        for small_batch in v_iter(iter_batch(s_batch, batch_size), verbose, message or "getting NN"):
            nn_dict_list.extend(self.get_knn_batch(small_batch, k))
        return nn_dict_list

    def get_knn_idxs_sub_batch(self, s_batch, k, batch_size=256, verbose=True, message=None):
        nn_dict_list = []
        for small_batch in v_iter(iter_batch(s_batch, batch_size), verbose, message or "getting NN Idxs"):
            nn_dict_list.extend(self.get_knn_idxs_batch(small_batch, k))
        return nn_dict_list
