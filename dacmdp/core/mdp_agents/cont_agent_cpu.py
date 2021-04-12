from lmdp.data.buffer import iter_batch
from collections import defaultdict
from sklearn.neighbors import KDTree
from collections import namedtuple
from tqdm import tqdm
from copy import deepcopy as cpy
import math
import random
# from lmdp.mdp.MDP_GPU import init2zero, init2list, init2dict, init2zero_def_dict, init2zero_def_def_dict
import time
import numpy as np
import heapq
from sklearn.neighbors import KDTree as RawKDTree
import torch
# from wrappers import *
from math import ceil 

MDPUnit = namedtuple('MDPUnit', 'tranProb origReward dist')
verbose_iterator = lambda it,vb: tqdm(it) if vb else it

import pickle as pk
from os import path

from typing import Dict, Any
import hashlib
import json

def init2dict():
    return {}

def init2list():
    return []


def init2zero():
    return 0


def init2zero_def_dict():
    return defaultdict(init2zero)


def init2zero_def_def_dict():
    return defaultdict(init2zero_def_dict)



# KD Tree helper function
class MyKDTree():
    def __init__(self, all_vectors):
        self.s2i, self.i2s = self._gen_vocab(all_vectors)
        self.KDtree = RawKDTree(np.array(list(self.s2i.keys())))

        self.get_knn = lambda s,k: self.get_knn_batch(np.array([s]), k)[0]
        self.get_nn = lambda s: list(self.get_knn_batch(np.array([s]), 1)[0])[0]
        self.get_nn_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_batch(s_batch,1)]

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

# DAC helper functions
def reward_logic(reward, dist, penalty_beta, penalty_type="linear"):
    if penalty_type == "none":
        disc_reward = reward
    elif penalty_type == "linear":
        disc_reward = reward - penalty_beta * dist
    else:
        assert False, "Unspecified Penalty type , please check parameters"
    return disc_reward

def kernel_probs(knn_dist_dict, delta):
    # todo Add a choice to do exponential averaging here.
    all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
    all_knn_probs = {nn: knn_kernel / sum(all_knn_kernels.values()) for nn, knn_kernel in
                     all_knn_kernels.items()}
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

    def __init__(self, seed_mdp, repr_model, build_args,solve_args , eval_args, action_space):

        # Main Components
        self.mdp_T = seed_mdp
        self.repr_model = repr_model
        self.build_args = build_args
        self.solve_args = solve_args
        self.eval_args = eval_args
        self.action_space = action_space
        self._verbose = False


        has_attributes = lambda v,a_list: all([hasattr(v, a) for a in a_list])

        assert has_attributes(repr_model, ["encode_action_batch", "encode_action_single","encode_state_batch",
                                 "encode_state_single", "predict_single_transition", "predict_batch_transition"])

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
        batch_iterator = verbose_iterator(iter_batch(range(len(buffer)), _batch_size), self._verbose)

        for idxs in batch_iterator:
            batch = buffer.sample_indices(idxs)
            batch_ob, batch_a, batch_ob_prime, batch_r, batch_nd = batch
            batch_d = 1 - batch_nd
            self._batch_parse(batch_ob, batch_a, batch_ob_prime, batch_r.view((-1,)), batch_d.view((-1,)))

        self.v_print("Step 1 [Parse Transitions]:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))
        self.v_print("length of to parse transitions", len(self.parsed_transitions))

    # Step 2
    def build_kdtree(self):
        """Builds KD tree on the states included in the parsed transitions"""
        self.v_print("Building kDTree"); st = time.time()

        assert self.parsed_transitions, "Empty Parsed Transitions"
        self.parsed_states = np.unique(np.stack([s for s,_,_,_,_ in self.parsed_transitions]),axis=0)
        self.s_kdTree = MyKDTree(self.parsed_states)
        
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
        self.tC = defaultdict(init2zero_def_def_dict)
        self.rC = defaultdict(init2zero_def_def_dict)

        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else "end_state"

        _batch_size = 256
        all_nn = []
        all_states = list(zip(*self.parsed_transitions))[0]

        for s_batch in verbose_iterator(iter_batch(all_states, _batch_size),self._verbose):
            all_nn.extend(self.s_kdTree.get_knn_batch(s_batch, self.build_args.tran_type_count))

        for i, tran in verbose_iterator(enumerate(self.parsed_transitions),self._verbose):
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
        self.mdp_T.s2i.update({s:i+2 for s,i in self.s_kdTree.s2i.items()})
        self.mdp_T.i2s.update({i+2:s for s,i in self.s_kdTree.s2i.items()})
        self.a2i = {a: i for i, a in enumerate(self.tran_types)}
        self.i2a = {i: a for i, a in enumerate(self.tran_types)}
        idx_missing = 0
        # todo account for filled_mask

        for s in verbose_iterator(self.tC,self._verbose):
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
    def process(self, train_buffer):
        """End to End processing of traing buffer with observations"""

        self.parse_all_transitions(train_buffer)
        self.build_kdtree()
        
        if self.build_args.rebuild_mdpfcache:
            self.v_print("Rebuilding MDP: loading Cached solution Vectors from",self.build_args.save_folder)
            self.load_sol_vectors(self.build_args.save_folder)
        else:
            self.intialize_dac_dynamics()
            self.initialize_MDP()
            self.solve_mdp()
            
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
        
    def load_sol_vectors(self,save_folder):
        
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
        
        assert all([hmap[a]==ld_hmap[a] for a in hash_attrs])
        assert all([mdp_hmap[a]==ld_mdp_hmap[a] for a in mdp_hash_attrs])
                    
        self.v_print("Initalization from cache complete")
        
        

    
    ## Logging Functions
    @property
    def mdp_distributions(self):
        rewards = self.mdp_T.rewardMatrix_cpu[:,:len(self.mdp_T.i2s),0].reshape(-1)

        all_distr = {"Transition Probabilty Distribution": 
                     self_loop_prob_distribution.mdp_T.tranProbMatrix_cpu[:,:len(self.mdp_T.i2s),0].reshape(-1),
                     "Reward Distribution": rewards[rewards>self.mdp_T.build_args.ur],
                     "Value Distribution": self.mdp_T.vD_cpu[:len(self.mdp_T.i2s)],
                     "Safe Value Distribution": self.mdp_T.s_vD_cpu[:len(self.mdp_T.i2s)],
                     "Q Value Distribution": None
                     # "NN distance Distribtuion": [d for d in self.dist_to_nn_cache if d != 0],
                     # "Self Loop Probability Distribution": mdp_T.self_loop_prob_distribution,
             }
        return all_distr 
    
    def log_all_mdp_metrics(self, mdp_frame_count, wandb_logger=None, tag_parent="MDP stats"):
        mdp_T = self.mdp_T
        all_distr = {"Transition Probabilty Distribution": mdp_T.tran_prob_distribution,
                     "Reward Distribution": mdp_T.reward_distribution,
                     "Value Distribution": list(mdp_T.valueDict.values()),
                     "Safe Value Distribution": list(mdp_T.s_valueDict.values()),
                     "State Action Fan In Distribution": mdp_T.state_action_fan_in_distribution,
                     "State Action Fan Out Distribution": mdp_T.state_action_fan_out_distribution,
                     "State Action Count Distribution": mdp_T.state_action_count_distribution,
                     # "NN distance Distribtuion": [d for d in self.dist_to_nn_cache if d != 0],
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
    
     # Step 3
    def intialize_dac_dynamics(self):
        """ Populates tC and rC based on the parsed transitions """
        self.v_print("----  Initializing Stochastic Dynamics  ----"); st = time.time()
        self.v_print("Step 3 [Populate Dynamics]: Running"); st = time.time()
        
        self.stt2a_idx_matrix = np.zeros((len(self.s_kdTree.s2i), self.build_args.tran_type_count)).astype(np.int32)
        
        self.orig_tD = defaultdict(init2zero_def_dict)
        self.tC = defaultdict(init2zero_def_def_dict)
        self.rC = defaultdict(init2zero_def_def_dict)

        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else "end_state"

        _batch_size = 256
        all_nn = []
        all_states = list(zip(*self.parsed_transitions))[0]
        
        for s_batch in verbose_iterator(iter_batch(all_states, _batch_size),self._verbose):
            all_nn.extend(self.s_kdTree.get_knn_batch(s_batch, self.build_args.tran_type_count))
        all_nn_s2i = {s:i for i,s in enumerate(all_states)}

        for i, tran in verbose_iterator(enumerate(self.parsed_transitions),self._verbose):
            s, a, ns, r, d = tran
            for tt, (nn_s, nn_d) in zip(self.tran_types, all_nn[i].items()):
                disc_r = reward_logic(r, nn_d, self.build_args.penalty_beta)
                nn_s_a = list(self.orig_tD[nn_s].keys())[0]
                self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]] = self.a_kdTree.s2i[nn_s_a]

                pred_ns =  list(self.orig_tD[nn_s].values())[0] # the prediction is simply the tran of nn_s_a
                
                if self.build_args.mdp_build_k == 1:
                    # skip calculating kNN
                    pred_ns_probs = {pred_ns:1}
                else:
                    pred_ns_dists = [all_nn[i+1].items()][:self.build_args.mdp_build_k]
                    pred_ns_probs = kernel_probs(all_nn_s2i[pred_ns], delta=self.build_args.knn_delta)
                    
                for pred_ns, prob in pred_ns_probs.items():
                    self.tC[s][tt][pred_ns] = int(prob*100)
                    self.rC[s][tt][pred_ns] = disc_r*int(prob*100)
            
        self.v_print("Step 3 [Populate Dynamics]: Complete,  Time Elapsed: {} \n\n".format(time.time() - st))
        
                

class DetPlcyEvalAgent(DeterministicAgent):
    def __init__(self, policy, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy = policy
        
        if self.build_args.tran_type_count != 1:
            print("Warining: tran_type_count must be 1 for policy Evaluation")
        
    
     # Step 3
    def intialize_dac_dynamics(self):
        """ Populates tC and rC based on the parsed transitions """
        self.v_print("----  Initializing Stochastic Dynamics  ----"); st = time.time()
        self.v_print("Step 3 [Populate Dynamics]: Running"); st = time.time()
        
        self.stt2a_idx_matrix = np.zeros((len(self.s_kdTree.s2i), self.build_args.tran_type_count)).astype(np.int32)
        
        self.orig_tD = defaultdict(init2zero_def_dict)
        self.tC = defaultdict(init2zero_def_def_dict)
        self.rC = defaultdict(init2zero_def_def_dict)

        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else "end_state"

        _batch_size = 256
        all_nn = []

        for t_batch in verbose_iterator(iter_batch(self.parsed_transitions, _batch_size),self._verbose):
            s_batch, a_batch, ns_batch, r_batch, d_batch = zip(*t_batch)
            all_nn.extend(self.s_kdTree.get_knn_batch(s_batch, self.build_args.tran_type_count))

        for i, tran in verbose_iterator(enumerate(self.parsed_transitions),self._verbose):
            s, a, ns, r, d = tran
            for tt, (nn_s, nn_d) in zip(self.tran_types, all_nn[i].items()):
                disc_r = reward_logic(r, nn_d, self.build_args.penalty_beta)
                nn_s_a, nn_ns = list(self.orig_tD[nn_s].keys())[0], list(self.orig_tD[nn_s].values())[0]

                self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]] = self.a_kdTree.s2i[nn_s_a]
                
                self.tC[s][tt][nn_ns] = 1
                self.rC[s][tt][nn_ns] = disc_r
            
        self.v_print("Step 3 [Populate Dynamics]: Complete,  Time Elapsed: {} \n\n".format(time.time() - st))