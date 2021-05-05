from lmdp.data.buffer import iter_batch
from collections import defaultdict
from sklearn.neighbors import KDTree
from collections import namedtuple
from tqdm import tqdm
from copy import deepcopy as cpy
import math
import random_policy
from lmdp.mdp.MDP_GPU import init2zero, init2list, init2dict, init2zero_def_dict, init2zero_def_def_dict
import time
import numpy as np
import heapq
from sklearn.neighbors import KDTree as RawKDTree
import torch
# from wrappers import *
from math import ceil 
from munch import munchify

# MDPUnit = namedtuple('MDPUnit', 'tranProb origReward dist')

import pickle as pk
from os import path

from typing import Dict, Any
import hashlib
import json


def verbose_iterator(iterator,
                     verbose, 
                     message = ""):
    if verbose:
        vb_iterator = tqdm(iterator)
        vb_iterator.set_description(message) 
    else:
        vb_iterator = iterator
        
    return vb_iterator


# KD Tree helper function
class MyKDTree():
    def __init__(self, all_vectors):
        self.s2i, self.i2s = self._gen_vocab(all_vectors)
        self.KDtree = RawKDTree(np.array(list(self.s2i.keys())))

        self.get_knn = lambda s,k: self.get_knn_batch(np.array([s]), k)[0]
        self.get_nn = lambda s: list(self.get_knn_batch(np.array([s]), 1)[0])[0]
        self.get_nn_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_batch(s_batch,1)]
        self.get_nn_sub_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_sub_batch(s_batch,1)]
        
        self.get_knn_idxs = lambda s,k: self.get_knn_idxs_batch(np.array([s]), k)[0]
        self.get_nn_idx = lambda s: list(self.get_knn_idxs_batch(np.array([s]), 1)[0])[0]
        self.get_nn_idx_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_batch(s_batch,1)]
        self.get_nn_idx_sub_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_idxs_batch(s_batch,1)]
        
        
    def _gen_vocab(self, all_vectors):
        s2i = {tuple(s):i for i,s in enumerate(all_vectors)}
        i2s = {i:tuple(s) for i,s in enumerate(all_vectors)}
        return s2i, i2s

    def get_knn_batch(self, s_batch, k):
        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        get_nn_dict = lambda dists, idxs: {self.i2s[int(idx)]: dist for dist, idx in zip(dists,idxs)}
        nn_dict_list = [get_nn_dict(dists,idxs) for dists, idxs in zip(dists_b, idxs_b)]
        return nn_dict_list
    
    def get_knn_idxs_batch(self, s_batch, k):
        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        get_nn_dict = lambda dists, idxs: {idx: dist for dist, idx in zip(dists,idxs)}
        nn_dict_list = [get_nn_dict(dists,idxs) for dists, idxs in zip(dists_b, idxs_b)]
        return nn_dict_list
    
    # Get knn with smaller batch sizes. | useful when passing large batches. 
    def get_knn_sub_batch(self, s_batch, k, batch_size = 256, verbose = True, message = None):
        nn_dict_list = []
        for small_batch in verbose_iterator(iter_batch(s_batch, batch_size), verbose, message or "getting NN"):
            nn_dict_list.extend(self.get_knn_batch(small_batch, k))
        return nn_dict_list 
    
    def get_knn_idxs_sub_batch(self, s_batch, k, batch_size = 256, verbose = True):
        nn_dict_list = []
        for small_batch in verbose_iterator(iter_batch(s_batch, batch_size), verbose, message or "getting NN Idxs"):
            nn_dict_list.extend(self.get_knn_idxs_batch(small_batch, k))
        return nn_dict_list
    

# DAC helper functions
def reward_logic(reward, dist, penalty_beta, penalty_type="linear"):
    if penalty_type == "none":
        disc_reward = reward
    elif penalty_type == "linear":
        disc_reward = reward - penalty_beta * dist
    else:
        assert False, "Unspecified Penalty type , please check parameters"
    return disc_reward

def kernel_probs(knn_dist_dict, delta, norm_by_dist=True):
    # todo Add a choice to do exponential averaging here.
    if norm_by_dist:
        all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
        all_knn_probs = {nn: knn_kernel / sum(all_knn_kernels.values()) for nn, knn_kernel in
                         all_knn_kernels.items()}
    else:
        all_knn_probs =  {s: 1/len(knn_dist_dict) for s,d in knn_dist_dict.items()}
        
    return all_knn_probs

def get_tran_types( tt_size):
    zero_matrix = torch.zeros((tt_size, tt_size), dtype=torch.float32, device="cpu")
    tt_tensor = zero_matrix.scatter_(1, torch.LongTensor(range(tt_size)).unsqueeze(1), 1).numpy()
    return [tuple(tt) for tt in tt_tensor]

# DAC Agent
class DeterministicAgent(object):
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """

    def __init__(self, seed_mdp, repr_model, build_args, solve_args , eval_args, action_space):

        # Main Components
        self.mdp_T = seed_mdp
        self.repr_model = repr_model
        self.build_args = build_args
        self.solve_args = solve_args
        self.eval_args = eval_args
        self.action_space = action_space
        self.action_len = action_space.shape[0]
        self.action_vec_size = action_space.shape[0]
        self._verbose = False


        has_attributes = lambda v,a_list: all([hasattr(v, a) for a in a_list])

        assert has_attributes(repr_model, ["encode_action_batch", "encode_action_single","encode_state_batch",
                                 "encode_state_single", "predict_next_state_single", "predict_next_state_batch"])

        assert has_attributes(build_args, ["mdp_build_k", "normalize_by_distance",
                                        "penalty_type", "penalty_beta", "knn_delta", "tran_type_count"])

        assert has_attributes(solve_args, ["gamma", "slip_prob"])

        assert has_attributes(eval_args, ["plcy_k", "soft_at_plcy"])

        # Main parameters
        self.s_kdTree = None
        self.parsed_transitions = []
        self.parsed_states = []
        self.tran_types = get_tran_types(self.build_args.tran_type_count)
        self.tt2i, self.i2tt = {tt:i for i,tt in enumerate(self.tran_types)}, {i:tt for i,tt in enumerate(self.tran_types)}
        self.stt2a_idx_matrix = None # Initialize after state KD Tree build

        self.seed_policies()


    # utility fxn
    def verbose(self):
        self._verbose = True
        return self

    def v_print(self,*args, **kwargs):
        if self._verbose: print(*args, **kwargs)

    # Step 1
    def _batch_parse(self, obs_batch, a_batch, obs_prime_batch, r_batch, d_batch):
        """Parses a observation transition to state transition and stores it in a to_commit list"""
        s_batch, s_prime_batch = map(self.repr_model.encode_state_batch, [obs_batch, obs_prime_batch])
        a_batch = self.repr_model.encode_action_batch(a_batch)
        r_batch = r_batch.cpu().numpy().astype(np.float32) 
        d_batch = d_batch.cpu().numpy()
        for s, a, s_prime, r, d in zip(s_batch, a_batch, s_prime_batch, r_batch, d_batch):
            self.parsed_transitions.append((s, a, s_prime, r, d))

    def _parse(self, obs, a, obs_prime, r, d):
        s, s_prime = map(self.repr_model.encode_state_batch, [obs, obs_prime])
        a = self.repr_model.encode_action_single(a)
        self.parsed_transitions.append((s, a, s_prime, r, d))

    def parse_all_transitions(self, buffer):
        """ Populates self.parsed_transitions using batch_parse function"""
        self.v_print("Step 1 (Parse Transitions):  Running");
        st = time.time()

        _batch_size = 256
        batch_iterator = verbose_iterator(iter_batch(range(len(buffer)), _batch_size), self._verbose, "Calculating latent repr from observations")

        for idxs in batch_iterator:
            batch = buffer.sample_indices(idxs)
            batch_ob, batch_a, batch_ob_prime, batch_r, batch_nd = batch
            batch_d = 1 - batch_nd
            self._batch_parse(batch_ob, batch_a, batch_ob_prime, batch_r.view((-1,)), batch_d.view((-1,)))

        self.state_vec_size = len(self.parsed_transitions[0][0])
        self.end_state_vector = tuple([404404]*self.state_vec_size)
        self.v_print("Step 1 [Parse Transitions]:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))
        self.v_print("length of to parse transitions", len(self.parsed_transitions))
        

    # Step 2
    def build_kdtree(self):
        """Builds KD tree on the states included in the parsed transitions"""
        self.v_print("Building kDTree"); st = time.time()

        assert self.parsed_transitions, "Empty Parsed Transitions"
        parsed_unique_states = np.unique(np.stack([s for s,_,_,_,_ in self.parsed_transitions] + [self.end_state_vector]),axis=0)
        self.s_kdTree = MyKDTree(parsed_unique_states)
        
        self.parsed_actions = np.unique(np.stack([a for s,a,_,_,_ in self.parsed_transitions]),axis=0)
        self.a_kdTree = MyKDTree(self.parsed_actions)

        self.v_print("kDTree build:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))

    # Step 3
    def intialize_dac_dynamics(self):
        """ Populates tC and rC based on the parsed transitions """
        self.v_print("Step 3 [Populate Dynamics]: Running"); st = time.time()
        
        if self.build_args.mdp_build_k > 1:
            print("Warning: behavior is undefined for mdp_build_k more than 1, proceed at your own risk")
        
        self.stt2a_idx_matrix = np.zeros((len(self.s_kdTree.s2i), self.build_args.tran_type_count)).astype(np.int32)
        
        self.orig_tD = defaultdict(init2zero_def_dict)
        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else self.end_state_vector

        
        self.tC = defaultdict(init2zero_def_def_dict)
        self.rC = defaultdict(init2zero_def_def_dict)
        
        # seed for end_state transitions
        for tt in self.tran_types:
            self.tC[self.end_state_vector][tt][self.end_state_vector] = 1
            self.rC[self.end_state_vector][tt][self.end_state_vector] = 0
        
        _batch_size = 256
        all_nn = []
        all_states = list(zip(*self.parsed_transitions))[0]

        for s_batch in verbose_iterator(iter_batch(all_states, _batch_size),self._verbose):
            all_nn.extend(self.s_kdTree.get_knn_batch(s_batch, self.build_args.tran_type_count))

        for i, tran in verbose_iterator(enumerate(self.parsed_transitions),self._verbose,  "Calculating DAC Dynamics"):
            s, a, ns, r, d = tran
            for tt, (nn_s, nn_d) in zip(self.tran_types, all_nn[i].items()):
                disc_r = reward_logic(r, nn_d, self.build_args.penalty_beta)
                nn_s_a, nn_ns = list(self.orig_tD[nn_s].keys())[0], list(self.orig_tD[nn_s].values())[0]

                self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]] = self.a_kdTree.s2i[nn_s_a]

                self.tC[s][tt][nn_ns] = 1
                self.rC[s][tt][nn_ns] = disc_r
            
        self.v_print("Step 3 [Populate Dynamics]: Complete,  Time Elapsed: {} \n\n".format(time.time() - st))
        
        
    # Step 4
    def initialize_MDP(self):
        """ Initializes the transtion Matrix in the internal MDP object """
        self.v_print("Step 4 [Initialize MDP]MM: Running"); st = time.time()

        # Add all to commit transitions to the MDP
        # track all to predict state action pairs
        assert len(self.s_kdTree.s2i) <= self.mdp_T.build_args.MAX_S_COUNT
        self.mdp_T.s2i.update({s:i for s,i in self.s_kdTree.s2i.items()})
        self.mdp_T.i2s.update({i:s for s,i in self.s_kdTree.s2i.items()})
        self.a2i = {a: i for i, a in enumerate(self.tran_types)}
        self.i2a = {i: a for i, a in enumerate(self.tran_types)}
        idx_missing = 0
        # todo account for filled_mask

        for s in verbose_iterator(self.tC,self._verbose,  "Writing DAC Dynamics to MDP"):
            for a in self.tC[s]:
                for slot, ns in enumerate(self.tC[s][a]):
                    # Get Indexes
                    try:
                        s_i, a_i, ns_i = self.mdp_T.s2i[s], self.a2i[a], self.mdp_T.s2i[ns]
                        # Get Counts
                        tran_count, r_sum = self.tC[s][a][ns], self.rC[s][a][ns]
                        self.mdp_T.update_count_matrices(s_i, a_i, ns_i, r_sum=r_sum, count=tran_count, slot=slot, append=False)
                    except: 
                        idx_missing  += 1
                self.mdp_T.update_prob_matrices(s_i, a_i)

        self.v_print("Step 4 [Initialize MDP]: Complete,  Time Elapsed: {}".format(time.time() - st))
        self.v_print(f"Missing Idx count:{idx_missing} \n\n")
        

    # step 5
    def solve_mdp(self):
        """ Solves the internal MDP object """
        self.v_print("Step 5 [Solve MDP]:  Running");st = time.time()

        self.mdp_T.curr_vi_error = 10
        self.mdp_T.solve(eps=0.001, mode="GPU", safe_bkp=True, verbose = self.verbose)
        # self.mdp_T.refresh_cache_dicts()

        # self.v_print("% of missing trans", self.mdp_T.unknown_state_action_count / (len(self.mdp_T.tD) * len(self.mdp_T.A)))
        self.v_print("Step 5 [Solve MDP]:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))

    # Main Function
    def process(self, train_buffer, match_hash = True):
        """End to End processing of traing buffer with observations"""

        self.cache_buffer =  train_buffer

        self.parse_all_transitions(train_buffer);time.sleep(1)
        self.build_kdtree();time.sleep(1)
        
        if self.build_args.rebuild_mdpfcache:
            self.v_print("Rebuilding MDP: loading Cached solution Vectors from",self.build_args.save_folder)
            self.load_sol_vectors(self.build_args.save_folder, match_hash = match_hash)
        else:
            self.intialize_dac_dynamics();time.sleep(1)
            self.initialize_MDP();time.sleep(1)
            self.solve_mdp();time.sleep(1)
            
        if self.build_args.save_mdp2cache:
            self.v_print("Caching MDP: Storing solution Vectors to",self.build_args.save_folder)
            self.cache_sol_vectors(self.build_args.save_folder)
            
    #### Value Functions #####
    def get_values_batch(self, s_batch, k=1):
        knnD_batch = self.s_kdTree.get_knn_batch(s_batch, k = k)
        knn_values = []
        for i, knnD in enumerate(knnD_batch):
            nn_idxs = [self.mdp_T.s2i[s] for s in knnD]
            knn_values.append(np.mean(self.mdp_T.vD_cpu[nn_idxs]))

        return knn_values

    def get_safe_values_batch(self, s_batch, k=1):
        knnD_batch = self.s_kdTree.get_knn_batch(s_batch, k = k)
        knn_values = []
        for i, knnD in enumerate(knnD_batch):
            nn_idxs = [self.mdp_T.s2i[s] for s in knnD]
            knn_values.append(np.mean(self.mdp_T.s_vD_cpu[nn_idxs]))

        return knn_values

    #### Policy Functions ####
    def random_policy(self, obs):
        return self.action_space.sample()

    def opt_policy(self, obs):
        return self.get_action_from_q_matrix(self.repr_model.encode_state_single(obs), self.mdp_T.qD_cpu,
                                             soft=self.eval_args.soft_at_plcy,
                                             weight_nn=self.build_args.normalize_by_distance,
                                             plcy_k=self.eval_args.plcy_k)

    def eps_optimal_policy(self, obs, epsilon = 0.1):
        return self.random_policy(obs) if (np.random.rand() < epsilon) else self.opt_policy(obs)

    def safe_policy(self, obs):
        return self.get_action_from_q_matrix(self.repr_model.encode_state_single(obs), self.mdp_T.s_qD_cpu,
                                             soft=self.eval_args.soft_at_plcy,
                                             weight_nn=self.build_args.normalize_by_distance,
                                             plcy_k=self.eval_args.plcy_k)

    def seed_policies(self):
        self.policies = {"optimal": self.opt_policy,
                         "random": self.random_policy,
                         "eps_optimal": self.eps_optimal_policy,
                         "safe": self.safe_policy}

    def sample_action_from_qval_dict(self, qval_dict):
        return random.choices(list(qval_dict.keys()), list(qval_dict.values()), k=1)[0]
    
    def get_action_from_q_matrix(self, hs, qMatrix, soft=False, weight_nn=False, plcy_k=1):
        qval_dict = {}
        knn_hs = self.s_kdTree.get_knn(hs, k=plcy_k)
        knn_hs_norm = kernel_probs(knn_hs, delta=self.build_args.knn_delta) \
            if weight_nn else {k: 1 / len(knn_hs) for k in knn_hs}

        for a in self.mdp_T.A:
            qval_dict[a] = np.sum([qMatrix[self.mdp_T.s2i[s], self.mdp_T.a2i[a]] * p for s, p in knn_hs_norm.items()])
    
        if soft:
            tt = self.sample_action_from_qval_dict(qval_dict)
        else:
            tt = max(qval_dict, key=qval_dict.get)
        
        nn_s,dist = list(knn_hs.items())[0]
        s = tuple(np.array(nn_s).astype("float32"))
        action = self.a_kdTree.i2s[self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]]]
        return np.array(action).reshape(-1,)

    ## Caching Functions
    def cache_sol_vectors(self,save_folder):
        hash_attrs = ["tt2i","i2tt"]
        mdp_hash_attrs = ["s2i", "a2i","A"]
        
        hmap = {a:dict_hash(getattr(self,a)) for a in hash_attrs}
        mdp_hmap = {a:dict_hash(getattr(self.mdp_T,a)) for a in mdp_hash_attrs}
        
        # Dump Logic Here 
        np.save(f"{save_folder}_stt2a_idx_matrix.npy",self.stt2a_idx_matrix)

        sol_mdp_matrices = ["vD_cpu","qD_cpu","s_vD_cpu","s_qD_cpu"]
        for m in sol_mdp_matrices:
            np.save(f"{save_folder}_{m}.npy",getattr(self.mdp_T, m))
                    
        pk.dump((hmap,mdp_hmap), open(f"{save_folder}_hmaps.pk","wb"))
        
        
        self.v_print("Solution vectors cached")
        
    def load_sol_vectors(self,save_folder, match_hash = True):
        
        # These attributes neeed to be initialized by 
        hash_attrs = ["tt2i","i2tt"]
        mdp_hash_attrs = ["s2i","a2i","A"]
        
        # Load mdp internal vars from build indx maps (indx to state)
        self.mdp_T.s2i.update({s:i+2 for s,i in self.s_kdTree.s2i.items()})
        self.mdp_T.i2s.update({i+2:s for s,i in self.s_kdTree.s2i.items()})
        
        # Load idx to value Matrices
        self.stt2a_idx_matrix = np.load(f"{save_folder}_stt2a_idx_matrix.npy")
                    
        sol_mdp_matrices = ["vD_cpu","qD_cpu","s_vD_cpu","s_qD_cpu"]
        for m in sol_mdp_matrices:
            setattr(self.mdp_T, m, np.load(f"{save_folder}_{m}.npy"))

        # verify hash maps of indx maps that has been rebuilt
        hmap = {a:dict_hash(getattr(self,a)) for a in hash_attrs}
        mdp_hmap = {a:dict_hash(getattr(self.mdp_T,a)) for a in mdp_hash_attrs}
        
        ld_hmap,ld_mdp_hmap = pk.load(open(f"{save_folder}_hmaps.pk","rb"))
        
        if match_hash:
            assert all([hmap[a]==ld_hmap[a] for a in hash_attrs])
            assert all([mdp_hmap[a]==ld_mdp_hmap[a] for a in mdp_hash_attrs]) 
        else:
            print("Warning Hash map of indexes are not being check, NN might be different")
                    
        self.v_print("Initalization from cache complete")
        
        

    
    ## Logging Functions
    @property
    def mdp_distributions(self):
        s_count = len(self.mdp_T.i2s)
        all_distr = {"Transition Probabilty Distribution": self.mdp_T.tranProbMatrix_cpu[:,:s_count,0].reshape(-1),
             "Reward Distribution": self.mdp_T.rewardMatrix_cpu[:,:s_count,0].reshape(-1),
             "Value Distribution": self.mdp_T.vD_cpu[:s_count].reshape(-1),
             "Safe Value Distribution": self.mdp_T.s_vD_cpu[:s_count].reshape(-1),
             }
        return all_distr 
    

                
def dict_hash(d: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.sha1()
    
    if isinstance(d,dict):
        d = {str(k): str(d[k]) for k in list(d)[::ceil(len(d)/1000)]}
        encoded = json.dumps(d).encode()
        
    elif isinstance(d,list):
        encoded = str(d[::ceil(len(d)/1000)]).encode()
        
    else:
        assert False, "Type not defined for encoding "
        
    dhash.update(encoded)
    return dhash.hexdigest()



class StochasticAgent(DeterministicAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.items4tt = lambda d: list(d.items())[:self.build_args.tran_type_count]
        self.keys4tt = lambda d: list(d.keys())[:self.build_args.tran_type_count]
        self.items4build_k = lambda d: list(d.items())[:self.build_args.mdp_build_k]
        self.keys4build_k = lambda d: list(d.keys())[:self.build_args.mdp_build_k]
        
        self.end_state_vector = None
        
        
    # Dataset Query functions
    # get action taken in that state or the next state from the given dataset. (assuming deterministic and unique states)
    def _query_action_from_D(self,s):
        return list(self.orig_tD[s].keys())[0]
    
    def _query_ns_from_D(self, s, force_vector = False):
        ns = list(self.orig_tD[s].values())[0]
        if force_vector and ns == self.end_state_vector:
            return self.end_state_vector
        else:
            return ns
    
    def _query_r_from_D(self, s, force_vector = False):
        return list(self.orig_rD[s].values())[0]
        
    
    
    # Main Functions | Override to change the nature of the MDP
    def get_candidate_actions(self, parsed_states):
        """ return candidate actiosn for all parsed_states  | numpy array with shape  [state_count, action_count, action_vec_size]"""
        self.v_print("Getting Candidate Actions [Start]"); st = time.time()
        
        parsed_s_candidate_actions = [[self._query_action_from_D(nn_s) for nn_s,d in self.items4tt(knn_dict)]
                                      for knn_dict in self.parsed_s_nn_dicts]
        
        self.v_print("Getting Candidate Actions [Complete],  Time Elapsed: {} \n".format(time.time() - st))
        return np.array(parsed_s_candidate_actions).astype(np.float32)
        
    def get_candidate_actions_dist(self, parsed_states, candidate_actions):
        """ return dists of all candidate actions  | numpy array with shape [state_count, action_count]"""
        self.v_print("Getting Candidate Action Distances"); st = time.time()
        
        parsed_s_candidate_action_dists = [[d for nn_s,d in self.items4tt(knn_dict)]  
                                           for knn_dict in self.parsed_s_nn_dicts]
        
        self.v_print("Getting Candidate Action Distances [Complete],  Time Elapsed: {} \n".format(time.time() - st))
        return np.array(parsed_s_candidate_action_dists).astype(np.float32)
        
    def get_candidate_predictions(self, parsed_states, candidate_actions):
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count, state_vec_size] """
        self.v_print("Getting predictions for given Candidate Actions"); st = time.time()
        parsed_s_candidate_predictions = [[self._query_ns_from_D(nn_s)  for nn_s,d in self.items4tt(knn_dict)]
                for knn_dict in self.parsed_s_nn_dicts]
        
        self.v_print("Getting predictions for given Candidate Actions [Complete],  Time Elapsed: {} \n\n".format(time.time() - st))
        return np.array(parsed_s_candidate_predictions).astype(np.float32)
    
    def get_candidate_rewards(self, parsed_states, candidate_actions):
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count] """
        self.v_print("Getting reward predictions for given Candidate Actions"); st = time.time()
        parsed_s_candidate_rewards = [[self._query_r_from_D(nn_s)  for nn_s,d in self.items4tt(knn_dict)]
                for knn_dict in self.parsed_s_nn_dicts]
        
        self.v_print("Getting reward predictions for given Candidate Actions [Complete],  Time Elapsed: {} \n\n".format(time.time() - st))
        return np.array(parsed_s_candidate_rewards).astype(np.float32)
    
    # Step 3
    def intialize_dac_dynamics(self):
        """ Populates tC and rC based on the parsed transitions """
        
        # HouseKeeping
        self.v_print(f"----  Initializing Stochastic Dynamics  NS Count {self.build_args.MAX_NS_COUNT}----"); st = time.time()
        self.v_print("Step 3 [Populate Dynamics]: Running"); st = time.time()
        
        if self.build_args.mdp_build_k != self.build_args.MAX_NS_COUNT:
            print("Warning Number of available ns slots and mdp build k is not the same. behavior is undefined")
        
        # Declare and Initialize helper variables
        self.stt2a_idx_matrix = np.zeros((len(self.s_kdTree.s2i), self.build_args.tran_type_count)).astype(np.int32)
        self.orig_tD, self.orig_rD = defaultdict(init2zero_def_dict), defaultdict(init2zero_def_dict)
        self.tC = defaultdict(init2zero_def_def_dict)
        self.rC = defaultdict(init2zero_def_def_dict)
        
        # seed for end_state transitions
        for tt in self.tran_types:
            self.tC[self.end_state_vector][tt][self.end_state_vector] = 1
            self.rC[self.end_state_vector][tt][self.end_state_vector] = 0

        # activates _query_action_from_D, and _query_ns_from_D
        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else self.end_state_vector
            self.orig_rD[s][a] = r 

        
        # calculate k for nearest neighbor lookups and helper functions
        nn_k = max(self.build_args.tran_type_count, self.build_args.mdp_build_k)
        
        # New NN variables
        self.parsed_states = list(zip(*self.parsed_transitions))[0]
        self.parsed_s_nn_dicts = self.s_kdTree.get_knn_sub_batch(self.parsed_states, nn_k, 
                                                                 batch_size = 256, verbose = self.verbose, 
                                                                 message= "NN for all parsed states")
        
        # candidate actions and predictions
        self.parsed_s_candidate_actions = self.get_candidate_actions(self.parsed_states) #  [state_count, action_count, action_vec_size] 
        self.parsed_s_candidate_action_dists = self.get_candidate_actions_dist(self.parsed_states, self.parsed_s_candidate_actions) # [state_count, action_count]
        self.parsed_s_candidate_predictions = self.get_candidate_predictions(self.parsed_states, self.parsed_s_candidate_actions) #  [state_count, action_count, state_vec_size]
        self.parsed_s_candidate_rewards = self.get_candidate_rewards(self.parsed_states, self.parsed_s_candidate_actions) 
        
        ac_size, acd_size, sp_size = self.parsed_s_candidate_actions.shape, self.parsed_s_candidate_action_dists.shape, self.parsed_s_candidate_predictions.shape
        self.parsed_s_candidate_predictions_knn_dicts = self.s_kdTree.get_knn_sub_batch(self.parsed_s_candidate_predictions.reshape(-1,self.state_vec_size),  self.build_args.mdp_build_k, 
                                                                               batch_size = 256, verbose = self.verbose,
                                                                              message = "NN for all predicted states.")
        
        assert len(self.parsed_s_candidate_actions[0]) == self.build_args.tran_type_count
        assert len(self.parsed_s_candidate_predictions[0]) == self.build_args.tran_type_count
        
        
        for s_idx, (s, a, ns, r, d) in verbose_iterator(enumerate(self.parsed_transitions),self._verbose, "Calculating DAC Dynamics"):    
            candidate_actions = self.parsed_s_candidate_actions[s_idx]
            candidate_action_dists = self.parsed_s_candidate_action_dists[s_idx]
            candidate_rewards = self.parsed_s_candidate_rewards[s_idx]
            
            for a_idx, (tt, cand_a, cand_d, cand_r) in enumerate(zip(self.tran_types, candidate_actions, candidate_action_dists,candidate_rewards)):
                preD_ns_idx = s_idx * len(self.tran_types) + a_idx
                # tt to action map 
                self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]] = self.a_kdTree.s2i[tuple(cand_a)]
                
                preD_ns_nn_dict_idx = s_idx*self.build_args.tran_type_count + a_idx
                pred_ns_nn_dict = {nn_s: d + cand_d for nn_s,d in self.parsed_s_candidate_predictions_knn_dicts[preD_ns_idx].items()}
                pred_ns_probs = kernel_probs(pred_ns_nn_dict, delta=self.build_args.knn_delta,
                                                norm_by_dist = self.build_args.normalize_by_distance)
                    
                # We are only concerned with transition counts in this phase. 
                # All transition counts will be properly converted to tran prob while inserting in MDP
                disc_r = reward_logic(cand_r, list(pred_ns_nn_dict.values())[0], self.build_args.penalty_beta)

                for dist, (pred_ns, prob) in zip(pred_ns_nn_dict.values(), pred_ns_probs.items()):
                     # reward discounted by the distance to state used for tt->a mapping. 
                    self.tC[s][tt][pred_ns] = int(prob*100)
                    self.rC[s][tt][pred_ns] = disc_r*int(prob*100)
            
        self.v_print("Step 3 [Populate Dynamics]: Complete,  Time Elapsed: {} \n\n".format(time.time() - st))

class StochasticAgentWithDelta(StochasticAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        

    # Main Functions | Override to change the nature of the MDP
    def get_candidate_predictions(self, parsed_states, candidate_actions):
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count, state_vec_size] """
        parsed_states = np.array(parsed_states)
        parsed_states_nn = np.array([self.keys4tt(knn_dict) for knn_dict in self.parsed_s_nn_dicts])
        parsed_states_nn_ns = np.array([np.array([self._query_ns_from_D(nn_s, force_vector= True) for nn_s,d in self.items4tt(knn_dict)])
                                        for knn_dict in self.parsed_s_nn_dicts])
        parsed_s_candidate_pred_vectors = np.expand_dims(parsed_states, 1) + parsed_states_nn_ns - parsed_states_nn

        return np.array(parsed_s_candidate_pred_vectors).astype(np.float32)


class StochasticAgentWithParametricPredFxn(StochasticAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    # Main Functions | Override to change the nature of the MDP
    def get_candidate_predictions(self,  parsed_states, candidate_actions):
        assert len(parsed_states) == len(candidate_actions)
        assert len(candidate_actions[0]) == self.build_args.tran_type_count
        assert len(candidate_actions[0][0]) == self.action_vec_size
        
        _batch_size = 256
        to_pred_s = np.repeat(parsed_states, self.build_args.tran_type_count, axis=0).reshape(-1, self.state_vec_size)
        to_pred_a = np.array(candidate_actions).reshape(-1, self.action_vec_size)
        
        batch_iterator = verbose_iterator(iterator = iter_batch(range(len(to_pred_s)), _batch_size), 
                                          verbose = self._verbose, 
                                          message = "Getting predictions using Dynamics model")
        
        parsed_pred_states = [self.repr_model.predict_next_state_batch(to_pred_s[idxs], to_pred_a[idxs]) for idxs in batch_iterator]      
        parsed_s_candidate_predictions = np.concatenate(parsed_pred_states)
        pred_shape = (-1,self.build_args.tran_type_count,self.state_vec_size)
        
        return np.array(parsed_s_candidate_predictions).astype(np.float32).reshape(pred_shape)
            
    
        
class StchPlcyEvalAgent(StochasticAgentWithParametricPredFxn):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        print("In StchPlcyEvalAgent")
        
        # Encoding Functions
        if self.build_args.tran_type_count != 1:
            print("Warining: tran_type_count must be 1 for policy Evaluation")
        
    # Main Functions | Override to change the nature of the MDP
    def get_candidate_actions(self, parsed_states):
        """ return candidate actiosn for all parsed_states  | numpy array with shape  [state_count, action_count, action_vec_size]"""
        _batch_size = 256
        batch_iterator = verbose_iterator(iterator = iter_batch(range(len(self.cache_buffer)), _batch_size), 
                                          verbose = self._verbose, 
                                         message = "Getting actions using policy model")
        parsed_pred_actions = [self.repr_model.predict_action_batch(self.cache_buffer.sample_indices(idxs)[0])
                                        for idxs in batch_iterator]
        parsed_s_candidate_actions = np.expand_dims(np.concatenate(parsed_pred_actions), 1)
        
        self.parsed_actions = np.unique(parsed_s_candidate_actions.reshape(-1,self.action_len),axis=0)
        self.a_kdTree = MyKDTree(self.parsed_actions)
        
        return parsed_s_candidate_actions
    
        
    def get_candidate_actions_dist(self, parsed_states, candidate_actions):
        print("Dummy action distance implemented: May need some further consideration")
        return np.zeros(self.parsed_s_candidate_actions.shape[:-1]).astype(np.float32)
        