from lmdp.data.buffer import get_iter_indexes
from collections import defaultdict
from sklearn.neighbors import KDTree
from collections import namedtuple
from tqdm import tqdm
from copy import deepcopy as cpy
import math
import random
from lmdp.mdp.MDP_GPU import init2zero, init2list, init2dict, init2zero_def_dict, init2zero_def_def_dict
import time
import numpy as np
import heapq

# from wrappers import *
MDPUnit = namedtuple('MDPUnit', 'tranProb origReward dist')

import pickle as pk
from os import path

verbose_iterator = lambda it,vb: tqdm(it) if vb else it

class SimpleAgent(object):
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """

    def __init__(self, mdp_T, net, fill_with, mdp_build_k, plcy_k=None,
                 kNN_on="stateAction", soft_at_plcy=False, normalize_by_distance=True,
                 penalty_type=False, penalty_beta=1, kNN_delta = 0.01
                 ):

        # Encoder network
        self.net = net
        assert all([hasattr(net, attr) for attr in
                    ["encode_action_batch", "encode_action_single","encode_state_batch", "encode_state_single", "predict_single_transition", "predict_batch_transition"]])

        # MDP build parameters
        self.mdp_T = mdp_T
        self.fill_with = fill_with
        self.mdp_build_k = mdp_build_k
        self.norm_by_dist = normalize_by_distance
        self.penalty_type = penalty_type
        self.penalty_beta = penalty_beta
        self.kNN_delta = kNN_delta

        # MDP policy parameters
        self.plcy_k = plcy_k or mdp_build_k
        self.soft_at_plcy = soft_at_plcy
        self.kNN_on = kNN_on

        # KNN Params
        self.known_tC = defaultdict(init2zero_def_def_dict)
        self.known_rC = defaultdict(init2zero_def_def_dict)

        # Abstraction Flags
        # self.abstraction_flag = abstraction_flag
        # self.abstraction_threshold = abstraction_threshold

        # internal vars
        self.nn_pairs = {}
        self.unseen_sa_pred_cache = {}  # predicted for unseen sa pairs
        self.in_mdp_sa_pairs = {}
        self.to_commit_sa_pairs = defaultdict(init2zero)
        self.to_commit_transitions = []
        self.dist_to_nn_cache = []
        self.mdp_cache = defaultdict(init2zero_def_def_dict)
        self.orig_reward_cache = defaultdict(init2zero_def_def_dict)
        self.iter = 0
        self.KDTree = None
        self.last_commit_iter = 0
        self.commit_seen_time, self.commit_predicted_time, self.solve_time = [], [], []

        self.seed_policies()

    def parse(self, obs, a, obs_prime, r, d):
        """
        Parses a observation transition to state transition and stores it in a to_commit list
        :param obs:
        :param a:
        :param obs_prime:
        :param r:
        :param d:
        :return:
        """
        # workhorse of MDP Agent
        # get corresponding states for the transition and add it to "to_commit" list
        s, s_prime = self.net.encode_state_single(obs), self.net.encode_state_single(obs_prime)
        a = self.net.encode_action_single(a)
        self.to_commit_transitions.append((s, a, s_prime, r, d))

    def batch_parse(self, obs_batch, a_batch, obs_prime_batch, r_batch, d_batch):
        s_batch, s_prime_batch = self.net.encode_state_batch(obs_batch), self.net.encode_state_batch(obs_prime_batch)
        a_batch = self.net.encode_action_batch(a_batch)
        for s, a, s_prime, r, d in zip(s_batch, a_batch, s_prime_batch, r_batch, d_batch):
            self.to_commit_transitions.append((s, a, s_prime, r, d))

    def commit_seen_transitions(self, verbose = False):
        # Add all to commit transitions to the MDP
        # track all to predict state action pairs
        if verbose: print("Len of to seed sa pairs", len(self.to_commit_transitions))
        iterator_ = tqdm(self.to_commit_transitions) if verbose else self.to_commit_transitions
        for s, a, s_prime, r, d in iterator_:
            self.mdp_T.consume_transition((s, a, s_prime, r, d))

            # cache for knn predictions
            s_prime = "end_state" if d else s_prime
            self.known_tC[s][a][s_prime] += 1
            self.known_rC[s][a][s_prime] += r


            for a_ in self.mdp_T.A:
                sa_pair = (s, a_)
                # 1 for seen sa_pair, 0 for unseen
                self.to_commit_sa_pairs[sa_pair] = 1 if a_ == a or self.to_commit_sa_pairs[sa_pair] == 1 else 0

                if not d and (s_prime, a_) not in self.to_commit_sa_pairs:
                    self.to_commit_sa_pairs[(s_prime, a_)] = 0

        self._update_nn_kd_tree()
        self._update_nn_kd_with_action_tree()
        if verbose: print("Len of to commit unseen sa pairs", len(self.to_commit_sa_pairs))

    def commit_predicted_transitions(self, verbose=False):
        if self.fill_with == "0Q_src-KNN":
            iterator_ = self.to_commit_sa_pairs.items()
            iterator_ = tqdm(iterator_) if verbose else iterator_

            for sa_pair, seen_flag in iterator_:
                # parse sa_pair
                s_, a_ = sa_pair
                s_i, a_i = self.mdp_T.s2i[s_], self.mdp_T.a2i[a_]

                # get distances
                knn_sa = self._get_knn_hs_kd_with_action_tree((s_, a_), k=self.mdp_build_k)
                knn_sa_normalized = self.get_kernel_probs(knn_sa, delta=self.kNN_delta)

                # houskeeping Code
                self.dist_to_nn_cache.extend(list(knn_sa.values()))
                nn_sa1, dist_1 = list(knn_sa.items())[0]
                self.nn_pairs[(s_,nn_sa1[0])] = dist_1

                # get new transition counts
                tran_counts, reward_counts = defaultdict(init2zero), defaultdict(init2zero)
                for nn_sa in knn_sa_normalized:
                    nn_s, a = nn_sa
                    norm_dist, dist = knn_sa_normalized[nn_sa], knn_sa[nn_sa]
                    for nn_ns in list(self.known_tC[nn_s][a].keys()):
                        orig_tc, orig_tr = self.known_tC[nn_s][a][nn_ns], self.known_rC[nn_s][a][nn_ns]
                        count_ = int(norm_dist * 100 * orig_tc) if self.norm_by_dist else 1
                        tran_counts[nn_ns] += count_
                        disc_reward = self.get_reward_logic(orig_tr / orig_tc, dist, self.penalty_type,
                                                            self.penalty_beta)
                        reward_counts[nn_ns] += count_ * disc_reward

                top_k_ns = heapq.nlargest(self.mdp_build_k, tran_counts, key=tran_counts.get)
                tran_counts = {s: tran_counts[s] for s in top_k_ns}  # filter for overflow
                reward_counts = {s: reward_counts[s] for s in top_k_ns}  # filter for overflow
                new_transitions = [(i, ns, tran_counts[ns], reward_counts[ns]) for i, ns in enumerate(tran_counts)]

                # update count matrices
                assert len(new_transitions) <= self.mdp_build_k, \
                    f"knn_len:{len(knn_sa)}, len: {len(new_transitions)}, tran_Counds: {len(tran_counts)}"
                for slot, ns, t_count, r_count in new_transitions:
                    ns_i = self.mdp_T.s2i[ns]
                    self.mdp_T.tranidxMatrix_cpu[a_i, s_i, slot] = ns_i
                    self.mdp_T.tranCountMatrix_cpu[a_i, s_i, slot] = t_count
                    self.mdp_T.rewardCountMatrix_cpu[a_i, s_i, slot] = r_count

                # update prob matrices
                self.mdp_T.update_prob_matrices(s_i, a_i)

            self.to_commit_sa_pairs = defaultdict(init2zero)
            self.to_commit_transitions = []

        elif self.fill_with == "none":
            print("Leaving the unknown  state actions ot the same state")
            pass
        else:
            assert False, "Fill with can only be with the model or knn"

    def get_reward_logic(self, reward, dist_to_nn_ns, penalty_type, penalty_beta):
        if penalty_type == "none":
            disc_reward = reward
        elif penalty_type == "linear":
            disc_reward = reward - penalty_beta * dist_to_nn_ns
        else:
            assert False, "Unspecified Penalty type , please check parameters"

        return disc_reward

    def solve_mdp(self, verbose = False):
        self.mdp_T.curr_vi_error = 10
        self.mdp_T.solve(eps=0.001, mode="GPU", safe_bkp=True, verbose = verbose)
        self.qvalDict_cache = cpy(self.mdp_T.qvalDict)
        self.valueDict_cache = cpy(self.mdp_T.valueDict)

    def get_value(self, o):
        return self.valueDict_cache[self.mdp_T._get_nn_hs_kdtree(self.net.encode_state_single(o))]

    def get_q_value(self, o, a_o):
        s, a = self.net.encode_state_single(o), self.net.encode_action_single(a_o)
        return self.qvalDict_cache[self.mdp_T._get_nn_hs_kdtree(s)][a]

    def build_mdp(self, train_buffer, verbose=False):

        if verbose: print("Step 1 (Parse Transitions):  Running")
        st = time.time()

        _batch_size = 256
        start_end_indexes = get_iter_indexes(len(train_buffer), _batch_size)
        iterator_ = tqdm(start_end_indexes) if verbose else start_end_indexes
        for start_i, end_i in iterator_:
            batch = train_buffer.sample_indices(list(range(start_i, end_i)))
            batch_ob, batch_a, batch_ob_prime, batch_r, batch_nd = batch
            batch_d = 1 - batch_nd
            self.batch_parse(batch_ob.numpy(), batch_a.numpy(), batch_ob_prime.numpy(),
                             batch_r.view((-1,)).numpy(), batch_d.view((-1,)).numpy())

        if verbose: print("Step 1 [Parse Transitions]:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))
        if verbose: print("Step 2 [Seed Seen Transitions + Unknown (s,a) pairs]:  Running")
        st = time.time()

        self.commit_seen_transitions(verbose=verbose)

        if verbose: print("Step 2 (Commit Seen Transitions):  Complete,  Time Elapsed: {} \n\n".format(time.time() - st))
        if verbose: print("Step 3 [Commit all Transitions]:  Running")
        st = time.time()

        self.commit_predicted_transitions(verbose=verbose)

        if verbose: print("Step 3 (Commit UnSeen Transitions):  Complete,  Time Elapsed: {}".format(time.time() - st))
        if verbose: print("Step 4 [Solve MDP]:  Running")
        st = time.time()

        self.solve_mdp()
        self.mdp_T.refresh_cache_dicts()
        self.seed_policies()

        if verbose: print("% of missing trans", self.mdp_T.unknown_state_action_count / (len(self.mdp_T.tD) * len(self.mdp_T.A)))
        if verbose: print("Step 4 [Solve MDP]:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))


    def log_all_mdp_metrics(self, mdp_frame_count, wandb_logger=None, tag_parent="MDP stats"):
        mdp_T = self.mdp_T
        all_distr = {"Transition Probabilty Distribution": mdp_T.tran_prob_distribution,
                     "Reward Distribution": mdp_T.reward_distribution,
                     "Value Distribution": list(mdp_T.valueDict.values()),
                     "Safe Value Distribution": list(mdp_T.s_valueDict.values()),
                     "State Action Fan In Distribution": mdp_T.state_action_fan_in_distribution,
                     "State Action Fan Out Distribution": mdp_T.state_action_fan_out_distribution,
                     "State Action Count Distribution": mdp_T.state_action_count_distribution,
                     "NN distance Distribtuion": [d for d in self.dist_to_nn_cache if d != 0],
                     "Self Loop Probability Distribution": mdp_T.self_loop_prob_distribution,
                     }

        all_scalars = {"State Count": len(mdp_T.tD)}

        if wandb_logger is not None:
            for name, metric in all_scalars.items():
                wandb_logger.log({tag_parent + "/_" + name: metric, 'mdp_frame_count': mdp_frame_count})

            for name, distr in all_distr.items():
                wandb_logger.log({tag_parent + "/Plotly_" + name: go.Figure(data=[go.Histogram(x=distr)]),
                                  'mdp_frame_count': mdp_frame_count})
                wandb_logger.log(
                    {tag_parent + "/_" + name: wandb_logger.Histogram(np.array(distr)),
                     'mdp_frame_count': mdp_frame_count})

        return all_distr

    
    # KD Tree helper functions    
    def _update_nn_kd_tree(self):
        assert len(self.mdp_T.s2i) > 2
        self.tree_states = [s for s in self.mdp_T.s2i if s not in self.mdp_T.omit_list]
        self.s_KDTree = KDTree(np.array(self.tree_states), leaf_size=40)
    
    def _update_nn_kd_with_action_tree(self):
        self.tree_f_states = {a: [] for a in self.mdp_T.A} # factore state list
        for s in self.mdp_T.known_sa_dict:
            for a in self.mdp_T.known_sa_dict[s]:
                self.tree_f_states[a].append(s)
    
        self.KDActionTrees = {a: KDTree(np.array(st_list), leaf_size=40) for a, st_list in self.tree_f_states.items() if
                              len(st_list) > 1}

    def _get_knn_hs_kd_with_action_tree(self, sa, k):
        s, a = sa
        if s in self.mdp_T.omit_list or not self.tree_f_states:
            return {sa: 0}
        no_action_available = a not in self.KDActionTrees
        a = next(iter(self.KDActionTrees)) if no_action_available else a
    
        nn_dist, nn_idx = self.KDActionTrees[a].query(np.array([s]), k=k)
        nn_dist, nn_idx = nn_dist.reshape(-1), nn_idx.reshape(-1)
        nn_dist += 1e12 if no_action_available else 0
        nn_dict = {(self.tree_f_states[a][int(idx)], a): nn_dist[i]
                   for i, idx in enumerate(nn_idx)}
        return nn_dict
    
    def _get_nn_hs_kd_with_action_tree(self, sa, return_dist=False):
        nn_sa, nn_dist = list(self._get_knn_hs_kd_with_action_tree(sa, k=1).items())[0]
        return (nn_sa, nn_dist) if return_dist else nn_sa
    
    def _get_knn_hs_kdtree(self, s, k):
        if s in self.omit_list or not self.state_list:
            return {s: 0}
    
        nn_dist, nn_idx = self.s_KDTree.query(np.array([s]), k=k)
        nn_dist, nn_idx = nn_dist.reshape(-1), nn_idx.reshape(-1)
        nn_dict = {self.tree_states[int(idx)]: nn_dist[i] for i, idx in enumerate(nn_idx)}
        return nn_dict
    
    def _get_nn_hs_kdtree(self, hs, return_dist=False):
        nn_hs, nn_dist = list(self._get_knn_hs_kdtree(hs, k=1).items())[0]
        return (nn_hs, nn_dist) if return_dist else nn_hs
    
    def get_kernel_probs(self, knn_dist_dict, delta=None):
        # todo Add a choice to do exponential averaging here.
        delta = delta or self.knn_delta
        all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
        all_knn_probs = {nn: knn_kernel / sum(all_knn_kernels.values()) for nn, knn_kernel in all_knn_kernels.items()}
        return all_knn_probs
    
    
    #### Policy Functions ####
    def get_eps_policy(self, greedy_policy, random_policy, epsilon=0.1):
        """
        returns a exploration exploitation policy based on epsilon , greedy and random policy
        """
        return lambda s: random_policy(s) if (np.random.rand() < epsilon) else greedy_policy(s)

    def random_policy(self, obs):
        return random.choice(self.mdp_T.A)

    def opt_policy(self, obs):
        return self.get_action_from_q_matrix(self.net.encode_state_single(obs), self.mdp_T.qD_cpu, 
                                      soft=self.soft_at_plcy,
                                      weight_nn=self.norm_by_dist,
                                      plcy_k=self.plcy_k, kNN_on=self.kNN_on)

    def safe_policy(self, obs):
        return self.get_action_from_q_matrix(self.net.encode_state_single(obs), self.mdp_T.s_qD_cpu, 
                                      soft=self.soft_at_plcy,
                                      weight_nn=self.norm_by_dist,
                                      plcy_k=self.plcy_k, kNN_on=self.kNN_on)

    def eps_optimal_policy(self, obs):
        eps_opt_pol = self.get_eps_policy(self.opt_policy, self.random_policy, epsilon=0.1)
        return eps_opt_pol(obs)

    def seed_policies(self, plcy_k=None, soft_at_plcy=None, kNN_on=None):
        self.plcy_k = plcy_k if plcy_k is not None else self.plcy_k
        self.soft_at_plcy = soft_at_plcy if soft_at_plcy is not None else self.soft_at_plcy
        self.kNN_on_sa = kNN_on if kNN_on is not None else self.kNN_on

        self.policies = {"optimal": self.opt_policy,
                         "random": self.random_policy,
                         "eps_optimal": self.eps_optimal_policy,
                         "safe": self.safe_policy}
        
    def sample_action_from_qval_dict(self, qval_dict):
        return random.choices(list(qval_dict.keys()), list(qval_dict.values()), k=1)[0]
    
    def get_action_from_q_matrix(self, hs, qMatrix, soft=False, weight_nn=False, plcy_k=1, kNN_on="stateAction"):
        qval_dict = {}
        if kNN_on == "stateAction":
            for a in self.mdp_T.A:
                knn_sa = self._get_knn_hs_kd_with_action_tree((hs, a), k=plcy_k)
                knn_sa_norm = self.get_kernel_probs(knn_sa, delta=self.kNN_delta) \
                    if weight_nn else {k: 1 / len(knn_sa) for k in knn_sa}
                qval_dict[a] = np.sum([qMatrix[self.mdp_T.s2i[sa[0]], self.mdp_T.a2i[sa[1]]] * p for sa, p in knn_sa_norm.items()])
        elif kNN_on == "stateActionRepr":
            knn_sa = self._get_knn_sa_repr_kdtree((hs,a), k=plcy_k)
            knn_sa_norm = self.get_kernel_probs(knn_sa, delta=self.kNN_delta) \
                if weight_nn else {k: 1 / len(knn_sa) for k in knn_sa}
            for a in self.mdp_T.A:
                qval_dict[a] = np.sum([qMatrix[self.mdp_T.s2i[sa[0]], self.mdp_T.a2i[sa[1]]] * p for s, p in knn_sa_norm.items()])
        else:
            knn_hs = self._get_knn_hs_kdtree(hs, k=plcy_k)
            knn_hs_norm = self.get_kernel_probs(knn_hs, delta=self.kNN_delta) \
                if weight_nn else {k: 1 / len(knn_hs) for k in knn_hs}
            for a in self.mdp_T.A:
                qval_dict[a] = np.sum([qMatrix[self.mdp_T.s2i[s], self.mdp_T.a2i[a]] * p for s, p in knn_hs_norm.items()])
    
        if soft:
            return self.sample_action_from_qval_dict(qval_dict)
        else:
            return max(qval_dict, key=qval_dict.get)
    

def get_eucledian_dist(s1, s2):
    return math.sqrt(sum([(s1[i] - s2[i]) ** 2 for i, _ in enumerate(s1)]))
