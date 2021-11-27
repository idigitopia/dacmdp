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
from sklearn.cluster import KMeans


# Project Specific Dependencies 
from lmdp.data.buffer import get_iter_indexes, iter_batch
from lmdp.mdp.MDP_GPU import init2zero, init2list, init2dict
from lmdp.mdp.MDP_GPU import init2zero_def_dict, init2zero_def_def_dict
from .disc_agent import dict_hash, v_iter, has_attributes, MyKDTree 
from .disc_agent import reward_logic, cost_logic, kernel_probs, get_one_hot_list, sample_random_action_gym
from .disc_agent import DACAgentBase


        

class DACAgentCont(DACAgentBase):
    """
    NN version Baseline for Continuous Actions. 
    get_candidate_actions : queries the action of the nearest neighbors.
    get_candidate_actions_dist : outputs the distance to the nearest neighbor for the corresponding action. 
    get_candidate_predictions : for the given candidate action, prediction is the seen next state for that particular action.
    get_candidate_rewards: for the given candidate action, predicted reward is the seen reward for the corresponding nn transition.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.action_len = self.action_space.shape[0]
        self.action_vec_size = self.action_space.shape[0]

        # Name of tran types. 
        self.tran_types = get_one_hot_list(self.build_args.tran_type_count)
        print("updated tran_types and indexing")
        
        # tran type name indexing variables.
        self.tt2i, self.i2tt = {tt:i for i,tt in enumerate(self.tran_types)}, {i:tt for i,tt in enumerate(self.tran_types)}
      
        # Dictionary filter functions
        self.items4tt = lambda d: list(d.items())[:self.build_args.tran_type_count]
        self.keys4tt = lambda d: list(d.keys())[:self.build_args.tran_type_count]
        self.items4build_k = lambda d: list(d.items())[:self.build_args.mdp_build_k]
        self.keys4build_k = lambda d: list(d.keys())[:self.build_args.mdp_build_k]
        
        # Step 2
    # Build KD Tree for parsed states. 
    def build_kdtree(self):
        """Builds KD tree on the states included in the parsed transitions"""
        assert self.parsed_transitions, "Empty Parsed Transitions"
        
        st = time.time()

        self.v_print("Building State KD Tree")
        # Compile a list of unique states parsed and add a end_state marker. 
        parsed_unique_states = np.unique(np.stack([s for s,_,_,_,_ in self.parsed_transitions] + [self.end_state_vector]),axis=0)
        # Build a KD Tree using the parsed States.
        self.s_kdTree = MyKDTree(parsed_unique_states)
        
        self.v_print("Building Action KD Tree")
        # Compile a list of unique actions parsed. 
        self.parsed_unique_actions = np.unique(np.stack([a for s,a,_,_,_ in self.parsed_transitions]),axis=0)
        # Build a KD Tree using the parsed actions.
        self.a_kdTree = MyKDTree(self.parsed_unique_actions)
        
        self.v_print("kDTree built:  Complete,  Time Elapsed: {}\n\n".format(time.time() - st))


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

    def get_candidate_predictions_knn_dicts(self, parsed_s_candidate_predictions):
        return self.s_kdTree.get_knn_sub_batch(parsed_s_candidate_predictions.reshape(-1, self.state_vec_size),
                                        self.build_args.mdp_build_k,
                                        batch_size=256, verbose=self.verbose,
                                        message="Calculating NN for all predicted states.")

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
        self.cC = defaultdict(init2zero_def_def_dict)
        
        # seed for end_state transitions
        for tt in self.tran_types:
            self.tC[self.end_state_vector][tt][self.end_state_vector] = 1
            self.rC[self.end_state_vector][tt][self.end_state_vector] = 0
            self.cC[self.end_state_vector][tt][self.end_state_vector] = 0
            
        # activates _query_action_from_D, and _query_ns_from_D
        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else self.end_state_vector
            self.orig_rD[s][a] = r 

        
        # calculate k for nearest neighbor lookups and helper functions
        nn_k = max(self.build_args.tran_type_count, self.build_args.mdp_build_k)
        
        # New NN variables
        self.parsed_states, self.parsed_actions,  self.parsed_next_states, self.parsed_rewards, self.parsed_terminals = \
            [np.array(v) for v in list(zip(*self.parsed_transitions))]
        self.parsed_s_nn_dicts = self.s_kdTree.get_knn_sub_batch(self.parsed_states, nn_k,
                                                                 batch_size = 256, verbose = self.verbose, 
                                                                 message= "Calculating NN for all parsed states")
        
        # candidate actions and predictions
        self.parsed_s_candidate_actions = self.get_candidate_actions(self.parsed_states) #  [state_count, action_count, action_vec_size] 
        self.parsed_s_candidate_action_dists = self.get_candidate_actions_dist(self.parsed_states, self.parsed_s_candidate_actions) # [state_count, action_count]
        self.parsed_s_candidate_predictions = self.get_candidate_predictions(self.parsed_states, self.parsed_s_candidate_actions) #  [state_count, action_count, state_vec_size]
        self.parsed_s_candidate_predictions_knn_dicts = self.get_candidate_predictions_knn_dicts(self.parsed_s_candidate_predictions)
        self.parsed_s_candidate_rewards = self.get_candidate_rewards(self.parsed_states, self.parsed_s_candidate_actions)

        # Sanity Check
        assert len(self.parsed_s_candidate_actions[0]) == self.build_args.tran_type_count
        assert len(self.parsed_s_candidate_predictions[0]) == self.build_args.tran_type_count
        assert len(self.parsed_s_candidate_action_dists[0]) == self.build_args.tran_type_count
        assert len(self.parsed_s_candidate_rewards[0]) == self.build_args.tran_type_count


        
        for s_idx, (s, a, ns, r, d) in v_iter(enumerate(self.parsed_transitions),self._verbose, "Calculating DAC Dynamics"):    
            candidate_actions = self.parsed_s_candidate_actions[s_idx]
            candidate_action_dists = self.parsed_s_candidate_action_dists[s_idx]
            candidate_rewards = self.parsed_s_candidate_rewards[s_idx]
            
            for a_idx, (tt, cand_a, cand_d, cand_r) in enumerate(zip(self.tran_types, candidate_actions, candidate_action_dists, candidate_rewards)):
                
                # tt to action map 
                self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]] = self.a_kdTree.s2i[tuple(cand_a)]
                
                preD_ns_idx = s_idx * len(self.tran_types) + a_idx
                pred_ns_nn_dict = {nn_s: d + cand_d for nn_s,d in self.parsed_s_candidate_predictions_knn_dicts[preD_ns_idx].items()}
                pred_ns_probs = kernel_probs(pred_ns_nn_dict, delta=self.build_args.knn_delta,
                                                norm_by_dist = self.build_args.normalize_by_distance)
                    
                # We are only concerned with transition counts in this phase. 
                # All transition counts will be properly converted to tran prob while inserting in MDP
                # Reward can be a function of distance of the nn of the prediction, or can also be accounted for individually. 
                cand_c = cost_logic(list(pred_ns_nn_dict.values())[0], self.build_args.penalty_beta)

                for dist, (pred_ns, prob) in zip(pred_ns_nn_dict.values(), pred_ns_probs.items()):
                    # reward discounted by the distance to state used for tt->a mapping. 
                    self.tC[s][tt][pred_ns] = int(prob*100)
                    self.rC[s][tt][pred_ns] = cand_r*int(prob*100)
                    self.cC[s][tt][pred_ns] = cand_c*int(prob*100)
            
        self.v_print("Step 3 [Populate Dynamics]: Complete,  Time Elapsed: {} \n\n".format(time.time() - st))



class DACAgentDelta(DACAgentCont):
    """
    NN version Baseline for Continuous Actions. 
    get_candidate_actions : queries the action of the nearest neighbors.
    get_candidate_actions_dist : outputs the distance to the nearest neighbor for the corresponding action. 
    get_candidate_predictions : for the given candidate action, prediction is the seen next state for that particular action.
    """

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

        
    def get_candidate_rewards(self, parsed_states, candidate_actions):
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count] """
        self.v_print("Getting reward predictions for given Candidate Actions"); st = time.time()
        parsed_s_candidate_rewards = [[self._query_r_from_D(nn_s)  for nn_s,d in self.items4tt(knn_dict)]
                for knn_dict in self.parsed_s_nn_dicts]
        
        self.v_print("Getting reward predictions for given Candidate Actions [Complete],  Time Elapsed: {} \n\n".format(time.time() - st))
        return np.array(parsed_s_candidate_rewards).astype(np.float32)


class DACAgentThetaDynamics(DACAgentCont):
    """
    DAC Agent with Parametric prediction Function. as Dynamics model.
    """
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
        
        batch_iterator = v_iter(iterator = iter_batch(range(len(to_pred_s)), _batch_size), 
                                          verbose = self._verbose, 
                                          message = "Getting predictions using Dynamics model")
        
        parsed_pred_states = [self.repr_model.predict_next_state_batch(to_pred_s[idxs], to_pred_a[idxs]) for idxs in batch_iterator]      
        parsed_s_candidate_predictions = np.concatenate(parsed_pred_states)
        pred_shape = (len(self.parsed_states),self.build_args.tran_type_count,self.state_vec_size)
        
        return np.array(parsed_s_candidate_predictions).astype(np.float32).reshape(pred_shape)
            

        
    def get_candidate_rewards(self, parsed_states, candidate_actions):
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count] """
        assert len(parsed_states) == len(candidate_actions)
        assert len(candidate_actions[0]) == self.build_args.tran_type_count
        assert len(candidate_actions[0][0]) == self.action_vec_size
        
        _batch_size = 256
        to_pred_s = np.repeat(parsed_states, self.build_args.tran_type_count, axis=0).reshape(-1, self.state_vec_size)
        to_pred_a = np.array(candidate_actions).reshape(-1, self.action_vec_size)
        
        batch_iterator = v_iter(iterator = iter_batch(range(len(to_pred_s)), _batch_size), 
                                          verbose = self._verbose, 
                                          message = "Getting Reward predictions using Reward model")
        
        parsed_pred_rewards = [self.repr_model.predict_reward_batch(to_pred_s[idxs], to_pred_a[idxs]) for idxs in batch_iterator]      
        parsed_s_candidate_rewards = np.concatenate(parsed_pred_rewards)
        
        return np.array(parsed_s_candidate_rewards).astype(np.float32).reshape((len(self.parsed_states),self.build_args.tran_type_count))


class DACAgentSARepr(DACAgentCont):
    """
    DAC Agent with Parametric prediction Function. as Dynamics model.
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    # Main Functions | Override to change the nature of the MDP
    # def get_candidate_actions(self, parsed_states): # Unchanged

    def get_candidate_actions_dist(self, parsed_states, candidate_actions):
        """ return dists of all candidate actions  | numpy array with shape [state_count, action_count]"""
        self.v_print("Getting distanceds for the given candidate actions");
        st = time.time()
        self.v_print("Getting Candidate Action Distances [Complete],  Time Elapsed: {} \n".format(time.time() - st))
        return np.zeros(self.parsed_s_candidate_actions.shape[:-1]).astype(np.float32)

    # Main Functions | Override to change the nature of the MDP
    def get_candidate_predictions(self,  parsed_states, candidate_actions):
        assert len(parsed_states) == len(candidate_actions)
        assert len(candidate_actions[0]) == self.build_args.tran_type_count
        assert len(candidate_actions[0][0]) == self.action_vec_size

        _batch_size = 256
        to_pred_s = np.repeat(parsed_states, self.build_args.tran_type_count, axis=0).reshape(-1, self.state_vec_size)
        to_pred_a = np.array(candidate_actions).reshape(-1, self.action_vec_size)

        batch_iterator = v_iter(iterator=iter_batch(range(len(to_pred_s)), _batch_size),
                                          verbose=self._verbose,
                                          message="Getting SA representation for candidate state actions using Dynamics model")

        self.parsed_s_candidate_sa_pairs = np.concatenate([self.repr_model.encode_state_action_batch(to_pred_s[idxs], to_pred_a[idxs]) for idxs in
                              batch_iterator])

        pred_shape = (len(self.parsed_states), self.build_args.tran_type_count, self.state_vec_size)
        return np.zeros(pred_shape)

    def get_candidate_predictions_knn_dicts(self, parsed_s_candidate_predictions):
        # make sa kd tree
        # search for knn s_idxs for all candidate sa representations.
        # find the ns fore each s_idxs.
        _batch_size = 256

        self.v_print("Making new kD tree with state action representation for given transitions")
        batch_iterator = v_iter(iterator=iter_batch(range(len(self.parsed_states)), _batch_size),
                                          verbose=self._verbose,
                                          message="Getting SA representation for parsed state and actions")

        self.parsed_sa_pairs = np.concatenate(
            [self.repr_model.encode_state_action_batch(self.parsed_states[idxs], self.parsed_actions[idxs]) for idxs in
             batch_iterator])

        self.sa_kdTree = MyKDTree(self.parsed_s_candidate_sa_pairs)



        self.candidate_sa_pairs_nn_idxs_dicts = self.sa_kdTree.get_knn_idxs_sub_batch(self.parsed_s_candidate_sa_pairs.reshape(-1, self.state_vec_size),
                                               self.build_args.mdp_build_k,
                                               batch_size=256, verbose=self.verbose,
                                               message="Calculating NN for all predicted states.")

        candidate_predictions_knn_dicts = [{self.parsed_next_states[nn_idx]:dist for nn_idx,dist in knn_dict} for knn_dict in self.candidate_sa_pairs_nn_idxs_dicts]
        return candidate_predictions_knn_dicts


    def get_candidate_rewards(self, parsed_states, candidate_actions): # Unchanged
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count] """
        self.v_print("Getting reward predictions for given Candidate Actions");
        st = time.time()
        parsed_s_candidate_rewards = [self.parsed_rewards[next(iter(knn_idx_dict))] for knn_idx_dict in self.parsed_s_nn_dicts]

        self.v_print("Getting reward predictions for given Candidate Actions [Complete],  Time Elapsed: {} \n\n".format(
            time.time() - st))
        return np.array(parsed_s_candidate_rewards).astype(np.float32)