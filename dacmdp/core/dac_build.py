from collections import defaultdict
from os import device_encoding
import os
import torch
import numpy as np
import gym 
from tqdm import tqdm
from functools import partial
from munch import Munch

from .models_action import ActionModelStore
from .models_sa_repr import ReprModelStore
from .utils_misc import viter_batch

from .utils_knn import THelper
from .dac_core import DACMDP, DACTransitionBatch


class DACBuildWithActionNames:
    def __init__(self, config, action_space, 
                action_model, repr_model,
                effective_batch_size = 1000, 
                batch_knn_engine =  THelper.batch_calc_knn_jit):
 
        self.effective_batch_size = effective_batch_size
        self.batch_knn_engine = batch_knn_engine
        self.batch_knn_idxs_engine = lambda b, D, k : self.batch_knn_engine(b, D, k)[0]
        
        ##################  Initial Setup. #############################################################
        self.device = config.mdpSolveArgs.device
        self.sa_repr_dim = config.reprModelArgs.repr_dim
        self.n_tran_types = config.mdpBuildArgs.n_tran_types
        self.action_space = action_space
        self.config = config
        self.action_dim = 1 if isinstance(action_space, gym.spaces.Discrete) else action_space.sample().shape[0]
        # self.buffer = buffer

        # We keep track of all_next_states because those are the ones being tracked in the MDP.
        # self.transitions = buffer.get_tran_tuples()
        self.all_states = torch.tensor([]).type(torch.FloatTensor)
        self.all_actions = torch.tensor([]).type(torch.FloatTensor)
        self.all_next_states = torch.tensor([]).type(torch.FloatTensor)
        self.A_names = torch.tensor([]).type(torch.FloatTensor)
        ################################################################################################

        ##################  DACMDP Core Setup. #########################################################
        self.dacmdp_core = DACMDP(n_tran_types=config.mdpBuildArgs.n_tran_types,
                            n_tran_targets=config.mdpBuildArgs.n_tran_targets,
                            sa_repr_dim=config.reprModelArgs.repr_dim,
                            penalty_beta = config.mdpBuildArgs.penalty_beta,
                            device=config.mdpSolveArgs.device,
                            penalty_type = config.mdpBuildArgs.penalty_type, 
                            batch_knn_engine = self.batch_knn_engine)
        ################################################################################################
        print("dacmdp_core_defined")


        #################. Filling up Dac Dynamics ###################################################
        self.action_model = action_model 
        self.repr_model = repr_model

        # Fetch Action Model
        if self.action_model:
            print("Using pre-initialized Action Model", self.action_model)
        else:
            print("Initializing Action Model")
            action_out_spec = (self.action_space, self.config.mdpBuildArgs.n_tran_types)
            action_model_store = ActionModelStore(action_out_spec, self.config, self.buffer)
            self.action_model = action_model_store.fetch(self.config.actionModelArgs.action_model_name)
            print("Initialized Action Model", self.action_model)

        # Fetch Repr Model
        if self.repr_model:
            print("Using pre-initialized Action Model", self.repr_model)
        else:
            print("Initializing Repr Model")
            repr_model_store = ReprModelStore(self.config, self.buffer)
            self.repr_model = repr_model_store.fetch(self.config.reprModelArgs.repr_model_name)
        
    
    def consume_transitions(self, transitions:DACTransitionBatch, 
                            replace_at_indices = None, 
                            batch_size = None, 
                            verbose = False,):
        #####  #####  Initialize Dacmdp_core tensors first  #####  #####  #####  #####  #####  #############################################
        self.dacmdp_core.init_transitions(transitions, replace_at_indices = replace_at_indices)
        nn, aa, tt = self.dacmdp_core.Ti.shape
        bb = transitions.states.size(0)
        batch_size = batch_size or bb
        print("nn after consumption, " , nn)
        #######################################################################################################################################
        
        
        #####  #####  core tensor update prep step  #####  #####  #####  #####  #####  #############################################
        if replace_at_indices is None :
            # Append Transitions
            self.all_states = torch.concat([self.all_states, transitions.states])
            self.all_actions = torch.concat([self.all_actions, transitions.actions])
            self.all_next_states = torch.concat([self.all_next_states, transitions.next_states])
            self.A_names = torch.concat([self.A_names, torch.zeros((bb,self.n_tran_types,self.action_dim))])
        else:
            self.all_states[replace_at_indices] = transitions.states
            self.all_actions[replace_at_indices] = transitions.actions
            self.all_next_states[replace_at_indices] = transitions.next_states
            self.A_names[replace_at_indices] = torch.zeros((bb,self.n_tran_types,self.action_dim))
        assert len(self.dacmdp_core.S) == len(self.all_next_states)
        assert len(self.dacmdp_core.S) == len(self.A_names)
        #######################################################################################################################################
        
        
        #####  ##### Update new indices or replaced indices  #####  #####  #####  #####  #####  #############################################
        u_indices = range(nn-bb, nn) if replace_at_indices is None else replace_at_indices
        ##################  self.fill_action_index_tensor(). ############################
        for state_indices in viter_batch(u_indices, batch_size, verbose = verbose, label = "Calculate Candidate Actions"):
            self.A_names[state_indices] = self.action_model.cand_actions_for_states(self.all_next_states[state_indices])
        ################################################################################################

        ##################  self.update_core_repr_tensors(). ############################
        for state_indices in viter_batch(u_indices, batch_size, verbose = verbose, label = "Calculate/Update Datsaet SA Representation"):
            self.calculate_n_set_dataset_representations(state_indices)

        for state_indices in viter_batch(u_indices, batch_size, verbose = verbose, label = "Calculate/Update Candidate Transition SA Representation"):
            self.calculate_n_set_transition_representations(state_indices)
        ################################################################################################

        ##################  self.update_core_dac_dynamics(). ############################
        for state_indices in viter_batch(u_indices, batch_size, verbose = verbose, label = "Update Transition model of core dacmdp"):
            self.dacmdp_core.update_tran_vectors(state_indices)
            
        ##################  self.update_core_dac_dynamics(). ############################
        if replace_at_indices is not None:
            affected_indices_tracker, placeholder_value = torch.zeros_like(self.dacmdp_core.V).cpu(), -(1e6)
            affected_indices_tracker[replace_at_indices] = placeholder_value
            affected_indices_mask = torch.min(torch.min(affected_indices_tracker[self.dacmdp_core.Ti], dim = -1).values, dim = -1).values == placeholder_value
            r_u_indices = torch.nonzero(affected_indices_mask).reshape(-1)
            for state_indices in viter_batch(r_u_indices, batch_size, verbose = verbose, label = "Update Transition model of stale states in dacmdp"):
                self.dacmdp_core.update_tran_vectors(state_indices) 
        #######################################################################################################################################
        
        
    def calculate_n_set_dataset_representations(self, state_indices):
        batch_size = len(state_indices)
        q_states, q_actions = self.all_states[state_indices], self.all_actions[state_indices]
        sa_reprs = self.repr_model.encode_state_action_pairs(q_states,q_actions)
        assert sa_reprs.shape == (batch_size, self.sa_repr_dim), (sa_reprs.shape , (batch_size, self.sa_repr_dim))
        self.dacmdp_core.set_dataset_representations(sa_reprs, state_indices)
    
    def calculate_n_set_transition_representations(self, state_indices):
        batch_size = len(state_indices)
        q_actions = self.A_names[state_indices].view((-1,self.action_dim))
        q_states = self.all_next_states[state_indices].repeat_interleave(self.n_tran_types, dim = 0)
        assert len(q_states) == len(q_actions)
        sa_reprs = self.repr_model.encode_state_action_pairs(q_states,q_actions)
        assert sa_reprs.shape == (batch_size*self.n_tran_types, self.sa_repr_dim)
        tran_repr_sets = sa_reprs.view((batch_size,self.n_tran_types,self.sa_repr_dim))
        self.dacmdp_core.set_transition_reprsentations(tran_repr_sets, state_indices)

    def to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).cuda()
        return x
    # policy liftup functions_nam
    def dummy_lifted_policy(self, s):
        nn_s_idx = self.batch_knn_idxs_engine(self.to_tensor(s).unsqueeze(0), self.dacmdp_core.S, 1)[0][0]
        policy_idx = self.dacmdp_core.Pi[nn_s_idx]
        action = self.A_names[nn_s_idx.cpu(),policy_idx.cpu()]
        return int(action.cpu().item()) if isinstance(self.action_space, gym.spaces.Discrete) else action

    def dac_nn_policy(self, s, policy_k = 5):
        # Encode the state.
        s = self.to_tensor(s)
        cand_actions = self.action_model.cand_actions_for_state(s)
        knn_s_idxs =  self.batch_knn_idxs_engine(s.unsqueeze(0).to(self.device), self.dacmdp_core.S, policy_k)[0]
        nn_mean_Q_vals = torch.mean(self.dacmdp_core.Q[knn_s_idxs], dim = 0)
        max_a_slot = torch.argmax(nn_mean_Q_vals)
        action = cand_actions[max_a_slot]
        return int(action.cpu().item()) if isinstance(self.action_space, gym.spaces.Discrete) else action

    def dac_lifted_policy(self,s):
        s = self.to_tensor(s)
        aa, tt = self.n_tran_types, self.dacmdp_core.dac_constants.n_tran_targets # number of Actions, number of targets for each action prediction.
        cand_actions = self.action_model.cand_actions_for_state(s) # is equal to the number of actions / transition types

        sa_reprs = self.repr_model.encode_state_action_pairs(s.repeat(self.n_tran_types).view(-1,len(s)).cpu(), cand_actions.cpu()).to(self.device) # SA Representations
        
        knn_idx_tensor, knn_dists_tensor = self.batch_knn_engine(sa_reprs, self.dacmdp_core.D_repr, k=tt)
        
        Tp = torch.nn.Softmax(dim = 1)(torch.log(1/(knn_dists_tensor+0.0001)))

        # Penalty 
        P = self.dacmdp_core.dac_constants.penalty_beta * knn_dists_tensor
        R_data = self.dacmdp_core.D_rewards[knn_idx_tensor.view(-1)].reshape(knn_idx_tensor.shape)

        R = R_data - P
        V = self.dacmdp_core.V[knn_idx_tensor.view(-1)].view((aa, tt)).to(self.device)
        C = self.dacmdp_core.C[knn_idx_tensor.view(-1)].view((aa, tt)).to(self.device)

        # import pdb; pdb.set_trace()
        # print(R.shape)
        Q_vals = torch.sum(Tp*(R+V), dim = 1).reshape(-1)
        C_vals = torch.sum(Tp*(P+C), dim = 1).reshape(-1)
        max_a_slot = torch.argmax(Q_vals)
        action = cand_actions[max_a_slot]

        return int(action.cpu().item())  if isinstance(self.action_space, gym.spaces.Discrete) else action

    #  helper function    
    def save(self, save_folder = None):
        save_folder = save_folder or self.config.mdpBuildArgs.save_folder
        self.dacmdp_core.save(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for t_name in ["A_names"]:
            s = f"torch.save(self.{t_name}, '{save_folder}/{t_name}.torch')"
            exec(s)
        print(f"Saved DACMDPAGENT to {save_folder}")

    def load(self, save_folder = None):
        self.dacmdp_core.load(save_folder)
        for t_name in ["A_names"]:
            exec(f"self.{t_name} = torch.load('{save_folder}/{t_name}.torch')")
        print(f"Loaded DACMDPAGENT from {save_folder}")


    def inspect_transitions(self,s_indx, show_state_repr = False):
        import numpy as np 
        np.set_printoptions(precision = 4, suppress=True)
        print(f"{'-'*60}")
        print(f"State Index {s_indx}")
        print(f"State Vector {self.transitions[s_indx][2]}")
        print(f"State appears as target in transition indexed: {s_indx}")
        print(f"State appears as source in transition indexed: {s_indx + 1}")

        ns_indxs_by_a_slot = self.dacmdp_core.Ti[s_indx]
        for a_slot, ns_indxs in enumerate(ns_indxs_by_a_slot):
            print(f"\t {'-'*40}")
            print(f"\t Action Slot: {a_slot}")
            print(f"\t Action Name: {self.A_names[s_indx][a_slot].numpy()}")
            print(f"\t Next State Indxs: {ns_indxs.cpu().numpy()}") 
            print(f"\t Next State Probs: {self.dacmdp_core.Tp[s_indx][a_slot].cpu().numpy()}") 
            print(f"\t Reward Distr: {self.dacmdp_core.R[s_indx][a_slot].cpu().numpy()}") 

            
            if show_state_repr:
                print(f"\t\tState Representations:")
                for idx in ns_indxs.cpu().numpy():
                    print(f"\t\t{str(idx).ljust(8)}: {self.dacmdp_core.D_repr[idx].cpu().numpy()}" )
                print("")
        print(f"{'-'*60}")
