# Default Python Pacvkages
import time
import heapq
from collections import namedtuple
from collections import defaultdict
import math
import random
from os import path
from typing import Dict, Any
import json 
import hashlib

# Standard Python Packages.
import torch
from torch import tensor
from tqdm import tqdm
from copy import deepcopy as cpy
from sklearn.neighbors import KDTree as RawKDTree
import pickle as pk
import numpy as np
from gym.spaces.discrete import Discrete


# Project Specific Dependencies 
from lmdp.data.buffer import get_iter_indexes, iter_batch
from lmdp.mdp.MDP_GPU import init2zero, init2list, init2dict
from lmdp.mdp.MDP_GPU import init2zero_def_dict, init2zero_def_def_dict


MDPUnit = namedtuple('MDPUnit', 'tranProb origReward dist')
verbose_iterator = lambda it,vb: tqdm(it) if vb else it
has_attributes = lambda v,a_list: all([hasattr(v, a) for a in a_list])


def get_eucledian_dist(s1, s2):
    return math.sqrt(sum([(s1[i] - s2[i]) ** 2 for i, _ in enumerate(s1)]))


def dict_hash(d: Dict[str, Any]) -> str:
    """returns an MD5 hash of a dictionary."""
    dhash = hashlib.sha1()

    if isinstance(d, dict):
        d = {str(k): str(d[k]) for k in list(d)[::math.ceil(len(d) / 1000)]}
        encoded = json.dumps(d).encode()

    elif isinstance(d, list):
        encoded = str(d[::math.ceil(len(d) / 1000)]).encode()

    else:
        assert False, "Type not defined for encoding "

    dhash.update(encoded)
    return dhash.hexdigest()

def v_iter(iterator, verbose, message = ""):
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


# KD Tree helper function
class MyKDTree():
    """
    Class to contain all the KD Tree related logics. 
    - Builds the index and inverseIndex for the vectors passed as the vocabulary for knn 
    - can get k/1 NN or k/1 NN of a batch of passed query vectors. 
    """
    def __init__(self, all_vectors):
        self.s2i, self.i2s = self._gen_vocab(all_vectors)
        self.KDtree = RawKDTree(np.array(list(self.s2i.keys())))

        self.get_knn = lambda s,k: self.get_knn_batch(np.array([s]), k)[0]
        self.get_nn = lambda s: list(self.get_knn_batch(np.array([s]), 1)[0])[0]
        self.get_nn_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_batch(s_batch,1)]
        self.get_nn_sub_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_sub_batch(s_batch,1)]
        
        self.get_knn_idxs = lambda s,k: self.get_knn_idxs_batch(np.array([s]), k)[0]
        self.get_nn_idx = lambda s: list(self.get_knn_idxs_batch(np.array([s]), 1)[0])[0]
        self.get_nn_idx_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_idxs_batch(s_batch,1)]
        self.get_nn_idx_sub_batch = lambda s_batch: [list(knnD)[0] for knnD in self.get_knn_idxs_sub_batch(s_batch,1)]
        
        
    def _gen_vocab(self, all_vectors):
        """
        generate index mappings and inverse index mappings. 
        """
        
        s2i = {tuple(s):i for i,s in enumerate(all_vectors)}
        i2s = {i:tuple(s) for i,s in enumerate(all_vectors)}
        return s2i, i2s

    def get_knn_batch(self, s_batch, k):
        """
        input: a list of query vectors. 
        output: a list of k-nn tuples for each query vector. 
        """

        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        get_nn_dict = lambda dists, idxs: {self.i2s[int(idx)]: dist for dist, idx in zip(dists,idxs)}
        nn_dict_list = [get_nn_dict(dists,idxs) for dists, idxs in zip(dists_b, idxs_b)]
        return nn_dict_list
    
    def get_knn_idxs_batch(self, s_batch, k):
        """
        input: a list of query vectors. 
        output: a list of k-nn idxs for each query vector.
        """
            
        s_batch = list(map(tuple, s_batch))
        dists_b, idxs_b = self.KDtree.query(np.array(s_batch), k=k)
        get_nn_dict = lambda dists, idxs: {idx: dist for dist, idx in zip(dists,idxs)}
        nn_dict_list = [get_nn_dict(dists,idxs) for dists, idxs in zip(dists_b, idxs_b)]
        return nn_dict_list
    
    # Get knn with smaller batch sizes. | useful when passing large batches. 
    def get_knn_sub_batch(self, s_batch, k, batch_size = 256, verbose = True, message = None):
        """
        # Get knn with smaller batch sizes. | useful when passing large batches.
        input: a large list of query vectors. 
        output: a large list of k-nn tuples for each query vector. 
        """
            
        nn_dict_list = []
        for small_batch in v_iter(iter_batch(s_batch, batch_size), verbose, message or "getting NN"):
            nn_dict_list.extend(self.get_knn_batch(small_batch, k))
        return nn_dict_list 
    
    def get_knn_idxs_sub_batch(self, s_batch, k, batch_size = 256, verbose = True, message = None):
        nn_dict_list = []
        for small_batch in v_iter(iter_batch(s_batch, batch_size), verbose, message or "getting NN Idxs"):
            nn_dict_list.extend(self.get_knn_idxs_batch(small_batch, k))
        return nn_dict_list
    
    @staticmethod
    def normalize_distances(knn_dist_dict, delta=None):
        # todo Add a choice to do exponential averaging here.
        delta = delta
        all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
        all_knn_probs = {nn: knn_kernel / sum(all_knn_kernels.values()) for nn, knn_kernel in all_knn_kernels.items()}
        return all_knn_probs

    
# DAC helper functions
def reward_logic(reward, dist, penalty_beta, penalty_type="linear"):
    """
    Returns discounted reward based on the given distance and penalty_beta.
    """
    disc_reward = reward - cost_logic(dist, penalty_beta, penalty_type)
    return disc_reward


def cost_logic(dist, penalty_beta, penalty_type="linear"):
    """
    Defines the logic of the cost. 
    """
    if penalty_type == "none":
        cost = 0
    elif penalty_type == "linear":
        cost = penalty_beta * dist
    else:
        assert False, "Unspecified Penalty type , please check parameters"
    return cost


def kernel_probs(knn_dist_dict, delta, norm_by_dist=True):
    """
    Return a dictionary with normalized distances. can now be treated as probabilities.
    Input norm_by_dist: returns a uniform distribution if False, if True Normalize. 
    """
    # todo Add a choice to do exponential averaging here.
    if norm_by_dist:
        all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
        all_knn_probs = {nn: knn_kernel / sum(all_knn_kernels.values()) for nn, knn_kernel in
                         all_knn_kernels.items()}
    else:
        all_knn_probs =  {s: 1/len(knn_dist_dict) for s,d in knn_dist_dict.items()}
        
    return all_knn_probs


def get_one_hot_list(tt_size):
    """
    Get list of one hot vectors. 
    input - tt_size: Size of the list. 
    Return - list of One Hot Vectors. 
    """
    zero_matrix = torch.zeros((tt_size, tt_size), dtype=torch.float32, device="cpu")
    tt_tensor = zero_matrix.scatter_(1, torch.LongTensor(range(tt_size)).unsqueeze(1), 1).numpy()
    return [tuple(tt) for tt in tt_tensor]

class CustomActionSpace():

    def __init__(self, action_list):
        self.action_list = action_list

    def sample(self,):
        return random.choice(self.action_list)



def get_action_list_from_space(action_space):
    """
    Returns a list of actions for the input gym space.
    """    
    if isinstance(action_space, Discrete):
        return [(i,) for i in list(range(0, action_space.n))]
    if isinstance(action_space, CustomActionSpace):
        return action_space.action_list
    else:
        print("No parse logic defined for Action Space" + str(type(action_space)))
        print("Warning Returning: None")
        return [None]
   
        
def sample_random_action_gym(action_space):
    if isinstance(action_space, Discrete):
        return [action_space.sample()]
    else:
        return action_space.sample()
    

# DAC Agent
class DACAgentBase(object):
    """
    DAC MDP Fromulation from the paper.
    # ToDo. Add more Description
    """

    def __init__(self, action_space, seed_mdp, repr_model, build_args, solve_args , eval_args):

        # Main Components
        self.action_space = action_space
        self.mdp_T = seed_mdp
        self.repr_model = repr_model
        self.build_args = build_args
        self.solve_args = solve_args
        self.eval_args = eval_args
        # self.action_len = action_space.shape[0]
        # self.action_vec_size = action_space.shape[0]
        self._verbose = False
        
        # Downstream Logic Placeholder
        self.policy_calls = {"base_pi":0, "mdp_pi":0}
        
        assert has_attributes(repr_model, ["encode_action_batch", "encode_action_single","encode_obs_batch",
                                 "encode_obs_single", "predict_next_state_single", "predict_next_state_batch"])

        assert has_attributes(build_args, ["mdp_build_k", "normalize_by_distance",
                                        "penalty_type", "penalty_beta", "knn_delta", "tran_type_count"])

        assert has_attributes(solve_args, ["gamma", "slip_prob"])

        assert has_attributes(eval_args, ["plcy_k", "soft_at_plcy"])

        # Main parameters
        self.s_kdTree = None
        self.parsed_states = []
        self.parsed_transitions = []
        
        # Tran Type parameters. 
        
        # Name of tran types. 
        self.tran_types = get_action_list_from_space(action_space)
        
        # tran type name indexing variables.
        self.tt2i, self.i2tt = {tt:i for i,tt in enumerate(self.tran_types)}, {i:tt for i,tt in enumerate(self.tran_types)}
        
        # tt index to action index mapping for each state. 
        #              tt-1  tt-2  tt-3
        #         s-1  a-i    . . .
        #         s-2  . 
        #         s-3  .
        self.stt2a_idx_matrix = None # Initialize after state KD Tree build, State transition type to idx matrix


    # utility fxn
    def verbose(self):
        """ 
        Set Verbose Flag
        """
        self._verbose = True
        return self

    def v_print(self,*args, **kwargs):
        """
        prints if verbose flag is set. 
        """
        if self._verbose: print(*args, **kwargs)

    # Step 1 
    # Convert the observation to states and parse transtitions. 
    def _parse(self, obs:torch.tensor, a:torch.tensor, obs_prime:torch.tensor, r:torch.tensor, d:torch.tensor):
        """
        Parses the transition and converts observations into states 
        using the repr_model.encode_obs_batch function. 
        stores the parsed transition with states into parsed_transitions arraay. 
        """ 
        
        s, s_prime = map(self.repr_model.encode_obs_batch, [obs, obs_prime])
        a = self.repr_model.encode_action_single(a)
        self.parsed_transitions.append((s, a, s_prime, r, d))

        
    def _batch_parse(self, obs_batch:torch.tensor, a_batch:torch.tensor, obs_prime_batch:torch.tensor, r_batch:torch.tensor, d_batch:torch.tensor):
        """
        Parses the transition batch and converts observations into states 
        using the repr_model.encode_obs_batch function. 
        stores the parsed transitions with states into parsed_transitions arraay. 
        """
        # s_batch and a_batch are expected to be tuples after encoding. input is torch arrays.  
        s_batch, s_prime_batch = map(self.repr_model.encode_obs_batch, [obs_batch, obs_prime_batch])
        a_batch = self.repr_model.encode_action_batch(a_batch)
        r_batch = r_batch.cpu().numpy().astype(np.float32) 
        d_batch = d_batch.cpu().numpy()
        for s, a, s_prime, r, d in zip(s_batch, a_batch, s_prime_batch, r_batch, d_batch):
            self.parsed_transitions.append((s, a, s_prime, r, d))


    def parse_all_transitions(self, buffer, _batch_size = 256):
        """ 
        Iteratively populates self.parsed_transitions using batch_parse function and the passed dataset/buffer 
        """
        
        self.v_print("Step 1 (Parse Transitions):  Running");
        st = time.time()

        batch_iterator = v_iter(iter_batch(range(len(buffer)), _batch_size), 
                                          self._verbose,
                                          "Calculating latent repr from observations")

        # Populate self.parsed_transitions
        for idxs in batch_iterator:
            batch = buffer.sample_indices(idxs)
            batch_ob, batch_a, batch_ob_prime, batch_r, batch_nd = batch
            batch_d = 1 - batch_nd
            self._batch_parse(batch_ob, batch_a, batch_ob_prime, batch_r.view((-1,)), batch_d.view((-1,)))

        self.state_vec_size = len(self.parsed_transitions[0][0])
        self.end_state_vector = tuple([404404]*self.state_vec_size)

        # Populate original transition dynamics. 
        self.orig_tD = defaultdict(init2zero_def_dict)
        self.orig_rD = defaultdict(init2zero_def_dict)
        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else self.end_state_vector
            self.orig_rD[s][a] = r if not d else 0

        self.v_print("Step 1 [Parse Transitions]:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))
        self.v_print("length of to parse transitions", len(self.parsed_transitions))
        
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

    # Step 2 
    # Build KD Tree for parsed states. 
    def build_kdtree(self):
        """Builds KD tree on the states included in the parsed transitions"""
        assert self.parsed_transitions, "Empty Parsed Transitions"
        
        st = time.time()

        self.v_print("Building State KD Tree")
        # Compile a list of unique states parsed and add a end_state marker. 
        self.parsed_unique_states = np.unique(np.stack([s for s,_,_,_,_ in self.parsed_transitions] + [self.end_state_vector]),axis=0)
        # Build a KD Tree using the parsed States.
        self.s_kdTree = MyKDTree(self.parsed_unique_states)
        
        self.v_print("Building Action KD Tree")
        # Compile a list of unique actions parsed. 
        self.parsed_unique_actions = np.unique(np.stack([a for s,a,_,_,_ in self.parsed_transitions]),axis=0)
        # Build a KD Tree using the parsed actions.
        self.a_kdTree = MyKDTree(self.parsed_unique_actions)
        
        # Build a state KD Tree for each action. #Discrete 
        self.v_print("Building State factored Action KD Tree")
        self.action_factored_states = {a: [] for a in get_action_list_from_space(self.action_space)} # factored state list
        for s, a, ns, r, d in self.parsed_transitions:
            self.action_factored_states[a].append(s)
                
        self.a_s_kdTrees = {a: MyKDTree(st_list) 
                            for a, st_list in self.action_factored_states.items()
                            if len(st_list) > 1}
                
        self.v_print("kDTree built:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))


    # Helper Function 1 for step 3
    # returns the canidate actions for all states. 
    # consider all available actions. (Default behavior)
    def get_candidate_actions(self, parsed_states):
        """ return candidate actiosn for all parsed_states  | numpy array with shape  [state_count, action_count, action_vec_size]"""
        self.v_print("Getting Candidate Actions [Start]"); st = time.time()
        
        parsed_s_candidate_actions = [get_action_list_from_space(self.action_space)
                                      for s in parsed_states]
        
        self.v_print("Getting Candidate Actions [Complete],  Time Elapsed: {} \n".format(time.time() - st))
        return np.array(parsed_s_candidate_actions).astype(np.float32)

    # Step 3
    # Calculate the DAC dynamics using the transitions and store it in form of transition counts. 
    def intialize_dac_dynamics(self):
        """ 
        Populates tC and rC based on the parsed transitions.
        """
            
        self.v_print("Step 3 [Populate Dynamics]: Running"); 
        st = time.time()
        
        # Each tt maps to action index | given the state. 
        self.stt2a_idx_matrix = np.zeros((len(self.s_kdTree.s2i), self.build_args.tran_type_count)).astype(np.int32)
        
        # Populate original transition dynamics. 
        self.orig_tD = defaultdict(init2zero_def_dict)
        self.orig_rD = defaultdict(init2zero_def_dict)
        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else self.end_state_vector
            self.orig_rD[s][a] = r if not d else 0


        # Populate the trancition, reward and cost matrices.
        self.tC = defaultdict(init2zero_def_def_dict)
        self.rC = defaultdict(init2zero_def_def_dict)
        self.cC = defaultdict(init2zero_def_def_dict)
        
        # seed for end_state transitions (self loop)
        for tt in self.tran_types:
            self.tC[self.end_state_vector][tt][self.end_state_vector] = 1
            self.rC[self.end_state_vector][tt][self.end_state_vector] = 0
            self.cC[self.end_state_vector][tt][self.end_state_vector] = 0
        
        _batch_size = 256 # Todo Make this a parameter. 
        # Get the ordered list of states in all parsed transitions. 
        # note that s_prime is not listed here. 
        # Todo see if this can be done as a map and only one call is made for each state.
        all_states = list(zip(*self.parsed_transitions))[0] 
        
        # Calculate nn for all candidate actions
        all_sa_nn = {}
        for tt, cand_a in zip(self.tran_types, get_action_list_from_space(self.action_space)):
            for s_batch in v_iter(iter_batch(all_states, _batch_size),self._verbose, 
                f"Calculate KNN for all candidate State action pairs for Tran Type {tt}"):
                knn_output = self.a_s_kdTrees[cand_a].get_knn_batch(s_batch, self.build_args.mdp_build_k)
                all_sa_nn.update({(state,cand_a):knn_dict for state, knn_dict in zip(s_batch, knn_output)})
                

        for i, tran in v_iter(enumerate(self.parsed_transitions),self._verbose,  "Calculating DAC Dynamics"):
            s, a, ns, r, d = tran
            
            # for each transition type / candidate action look at the nearest neighbor transitions seen in the dataset
            for tt, cand_a in zip(self.tran_types,get_action_list_from_space(self.action_space)):
                self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]] = self.a_kdTree.s2i[cand_a]
                knn_cand_sa = all_sa_nn[(s,cand_a)]
                knn_cand_sa_normalized = MyKDTree.normalize_distances(knn_cand_sa, delta=self.build_args.knn_delta)
                
                # nn state where the candidate action is taken.
                for (nn_s, nn_d), (_, nn_prob) in zip(knn_cand_sa.items(), knn_cand_sa_normalized.items()):
                    # cost for including the given candidate action.
                    cost = cost_logic(nn_d, self.build_args.penalty_beta)
                    # expected reward for the given candidate action.
                    # look at the reward for given candidate action on the nn_s.
                    expec_r = self.orig_rD[nn_s][cand_a]
                    nn_ns = self.orig_tD[nn_s][cand_a]

                    # fill in the tran counts.
                    count_ = int(nn_d * 100) if self.build_args.normalize_by_distance else 1
                    self.tC[s][tt][nn_ns] += count_
                    self.rC[s][tt][nn_ns] += r
                    self.cC[s][tt][nn_ns] += cost
                
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
        self.mdp_T.a2i = self.tt2i
        self.mdp_T.i2a = self.i2tt
        idx_found, idx_missing = 0, 0
        # todo account for filled_mask

        for s in v_iter(self.tC,self._verbose,  "Writing DAC Dynamics to MDP"):
            for tt in self.tC[s]:
                for slot, ns in enumerate(self.tC[s][tt]):
                    try:
                        # Get Indexes
                        s_i, tt_i = self.mdp_T.s2i[s], self.mdp_T.a2i[tt] 
                        ns_i = self.mdp_T.s2i[ns]
                    
                    except KeyError: 
                        idx_missing  += 1

                    # Get Counts
                    tran_count, r_sum, c_sum  = self.tC[s][tt][ns], self.rC[s][tt][ns], self.cC[s][tt][ns]

                    idx_found += 1

                    

                    if self.mdp_T.is_factored:
                        self.mdp_T.update_count_matrices(s_i, tt_i, ns_i, r_sum=r_sum, c_sum=c_sum, count=tran_count, slot=slot, append=False)
                    else:
                        self.mdp_T.update_count_matrices(s_i, tt_i, ns_i, r_sum=r_sum - c_sum, count=tran_count, slot=slot, append=False)
                    
                    
                        
                self.mdp_T.update_prob_matrices(s_i, tt_i)

        self.v_print("Step 4 [Initialize MDP]: Complete,  Time Elapsed: {}".format(time.time() - st))
        self.v_print(f"Total index processed:{idx_found + idx_missing}, Success: {idx_found} Failure: {idx_missing} | {100*idx_missing/(idx_found + idx_missing):.4f}% \n\n")
        
    # step 5
    def solve_mdp(self):
        """ Solves the internal MDP object """
        self.v_print("Step 5 [Solve MDP]:  Running");st = time.time()

        self.mdp_T.curr_vi_error = 10
        self.mdp_T.solve(eps=0.001, mode="GPU", safe_bkp=True, verbose = self.verbose)
        # self.mdp_T.refresh_cache_dicts()

        self.v_print("Step 5 [Solve MDP]:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))
        self.v_print("Seeding policies")
        self.seed_policies()

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
        return sample_random_action_gym(self.action_space)

    def opt_policy(self, obs):
        return self.get_action_from_q_matrix(self.repr_model.encode_obs_single(obs), self.mdp_T.qD_cpu,
                                             soft=self.eval_args.soft_at_plcy,
                                             weight_nn=self.build_args.normalize_by_distance,
                                             plcy_k=self.eval_args.plcy_k)

    def eps_optimal_policy(self, obs, epsilon = 0.1):
        return self.random_policy(obs) if (np.random.rand() < epsilon) else self.opt_policy(obs)

    def safe_policy(self, obs):
        return self.get_action_from_q_matrix(self.repr_model.encode_obs_single(obs), self.mdp_T.s_qD_cpu,
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
    

    def get_action_from_q_matrix(self, s, qMatrix, soft=False, weight_nn=False, plcy_k=1):
        """
        Fetches the optimal action from the q_matrix for the query state. 
        ToDo this needs some additional work. 
        """
        knn_s = self.s_kdTree.get_knn(s, k=plcy_k)
        knn_s_norm = kernel_probs(knn_s, delta=self.build_args.knn_delta) \
            if weight_nn else {k: 1 / len(knn_s) for k in knn_s}

        qval_dict = defaultdict(lambda:[])
        for s, nn_d in knn_s.items():
            for tt in self.tran_types:
                s_i = self.s_kdTree.s2i[s] # get state index
                tt_i = self.tt2i[tt] # get tran type index
                a_i = self.stt2a_idx_matrix[s_i][tt_i] # get action index from (s_i,tt_i) to a_i map
                a_vec = self.a_kdTree.i2s[a_i] # get action vector.
                
                cost = cost_logic(nn_d, self.build_args.penalty_beta)
                qval_dict[a_vec].append(qMatrix[s_i, tt_i] - cost)
        
        avg_qval_dict = {a_vec: np.mean(vals) for a_vec, vals in qval_dict.items()}
    
        if soft:
            a_vec = self.sample_action_from_qval_dict(avg_qval_dict)
        else:
            a_vec = max(avg_qval_dict, key=avg_qval_dict.get)
        
        # nn_s,dist = list(knn_s.items())[0]
        # s = tuple(np.array(nn_s).astype("float32"))
        # action = self.a_kdTree.i2s[self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]]]
        return np.array(a_vec).reshape(-1,)


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
