from os import device_encoding
import os
import torch
from tqdm import tqdm
from functools import partial
from munch import Munch

from .utils_knn import THelper
# from .utils_misc import tensor_set_minus
from dataclasses import dataclass

@dataclass
class DACTransitionBatch:
    states: torch.FloatTensor
    actions: torch.FloatTensor
    next_states: torch.FloatTensor
    rewards: torch.FloatTensor
    terminals: torch.LongTensor

    @classmethod
    def combine_transition_batches(cls, list_of_transition_batches):
        # Check if the list is empty
        if not list_of_transition_batches:
            raise ValueError("The list of transition batches cannot be empty")

        # Initialize combined attributes with the first batch to ensure correct shapes and types
        combined_states = list_of_transition_batches[0].states
        combined_actions = list_of_transition_batches[0].actions
        combined_next_states = list_of_transition_batches[0].next_states
        combined_rewards = list_of_transition_batches[0].rewards
        combined_terminals = list_of_transition_batches[0].terminals

        # Iterate over the remaining batches and concatenate them
        for batch in list_of_transition_batches[1:]:
            combined_states = torch.cat([combined_states, batch.states], dim=0)
            combined_actions = torch.cat([combined_actions, batch.actions], dim=0)
            combined_next_states = torch.cat([combined_next_states, batch.next_states], dim=0)
            combined_rewards = torch.cat([combined_rewards, batch.rewards], dim=0)
            combined_terminals = torch.cat([combined_terminals, batch.terminals], dim=0)

        # Return a new DACTransitionBatch instance with the combined data
        return cls(states=combined_states, actions=combined_actions, next_states=combined_next_states,
                   rewards=combined_rewards, terminals=combined_terminals)



class DACMDP_CORE():

    ######################## Nomenclature ########################################
    # nn, aa, tt = self.S.len, n_tran_types, n_tran_targets
    # self.Tp = torch.zeros((nn, aa, tt)).type(torch.float32).to(device=device)
    # self.Ti = torch.zeros((nn, aa, tt)).type(torch.LongTensor).to(device=device)
    # self.R = torch.zeros((nn, aa, tt)).type(torch.float32).to(device=device)
    # self.Tdist = torch.zeros((nn, aa, tt)).type(torch.float32).to(device=device) # Distances for s,a,s' approximations
    # self.P = torch.zeros((nn, aa, tt)).type(torch.float32).to(device=device) # Penalites for s,a,s' approximations
    # self.Q = torch.zeros((nn, aa)).type(torch.float32).to(device=device)
    # self.V = torch.zeros((nn,)).type(torch.float32).to(device=device)
    # self.C = torch.zeros((nn,)).type(torch.float32).to(device=device) # Cost of optimal policy for each state.
    # self.Pi = torch.zeros((nn)).type(torch.LongTensor).to(device=device)
    # self.V_safe = torch.zeros((nn,)).type(torch.float32).to(device=device)
    # self.Pi_safe = torch.zeros((nn)).type(torch.LongTensor).to(device=device)
    ##############################################################################

    ########################### Rabbit King ####################################
    @torch.jit.script
    def bellman_backup_operator(Ti, Tp, R, P, Q, V, gamma):
        R=R-P
        Q = torch.sum(torch.multiply(R, Tp), dim=2) + \
            gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
        max_obj = torch.max(Q, dim=1)
        V_prime, Pi = max_obj.values, max_obj.indices
        epsilon = torch.max(V_prime-V)
        return epsilon, Q, Pi, V_prime
    ############################################################################

    @torch.jit.script
    def dac_bellman_backup_operator(Ti, Tp, R, P, Q, V, C, gamma):
        NS_V = gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2) # (nn,aa)
        NS_C = gamma[0]*torch.sum(torch.multiply(Tp, C[Ti]), dim=2) # (nn,aa)
        
        Q_reward = torch.sum(torch.multiply(R, Tp), dim=2) # (nn,aa)
        Q_penalty = torch.sum(torch.multiply(P, Tp), dim=2) # (nn,aa)
        Q = Q_reward - Q_penalty + NS_V # (nn,aa)
        
        max_obj = torch.max(Q, dim=1)
        V_prime, Pi = max_obj.values, max_obj.indices # (nn,) (nn,)
        epsilon = torch.max(V_prime-V)
        
        nn,tt,aa = Ti.shape
        C_prime = (Q_penalty + NS_C).view(-1)[max_obj.indices + torch.arange(0,nn*tt,tt).to("cuda")]
        return epsilon, Q, Pi, V_prime, C_prime
            
    @torch.jit.script
    def dac_safe_bellman_backup_operator(Ti, Tp, R, P, Q, V, C, gamma):
        NS_V = gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2) # (nn,aa)
        NS_C = gamma[0]*torch.sum(torch.multiply(Tp, C[Ti]), dim=2) # (nn,aa)
        
        Q_reward = torch.sum(torch.multiply(R, Tp), dim=2) # (nn,aa)
        Q_penalty = torch.sum(torch.multiply(P, Tp), dim=2) # (nn,aa)
        Q = Q_reward - Q_penalty + NS_V # (nn,aa)
        
        max_obj = torch.max(Q, dim=1) 
        V_star, Pi = max_obj.values, max_obj.indices # (nn,) (nn,)
        V_prime= 0.9*V_star + 0.1 * torch.mean(Q, dim = 1)
        epsilon = torch.max(V_prime-V)
        
        nn,tt,aa = Ti.shape
        C_prime = (Q_penalty + NS_C).view(-1)[max_obj.indices + torch.arange(0,nn*tt,tt).to("cuda")]
        return epsilon, Q, Pi, V_prime, C_prime

    # Designed to be updatable on the fly.
    # Entities to think about.
    # S as a current Dataset State space for a compressed representation of the observation.
    # A as a current Dataset Action space for a 1-1 representation of the actions available.
    # SA as a current Dataset State-Action space for holding the representations for all sa pairs in the datset.
    # i.e. change the state vectors of certain indices.
    # i.e. change the candidate_action and sa_repr of

    # Transitions (add one at a time. )
    # update D_terminal_indices while adding a transition. 
    # Update S for each added transition. 
    # Update self.nn aa and tt 
    # Update D_rewards
    # Update all Ti Tp vectors. 

    def __init__(self, n_tran_types, n_tran_targets, sa_repr_dim, penalty_beta = 1, device='cuda', penalty_type = "linear",
                    batch_calc_knn_ret_flat_engine = THelper.batch_calc_knn_ret_flat_pykeops):
        # ToDo Some sanity checkes for transitions

        super().__init__()

        self.batch_calc_knn_ret_flat_engine = batch_calc_knn_ret_flat_engine 
        self.device = device

        self.dac_constants = Munch()
        self.dac_constants.n_tran_types = n_tran_types
        self.dac_constants.n_tran_targets = n_tran_targets
        self.dac_constants.sa_repr_dim = sa_repr_dim 
        self.dac_constants.penalty_beta = penalty_beta
        self.dac_constants.penalty_type = penalty_type

        # Dataset Caches
        self.D_rewards = torch.FloatTensor([]).to("cuda")
        self.D_terminals = torch.LongTensor([]).to("cuda")
        self.D_terminal_indices = torch.LongTensor([]).to("cuda")
        self.D_repr = torch.FloatTensor([]).to("cuda")

        # Core States
        self.S = torch.FloatTensor([]).to("cuda")
        self.S.index = partial(THelper.lookup_index_by_hash, torch_matrix=self.S)
        self.S.nn_index = partial(THelper.lookup_index_by_hash, torch_matrix=self.S)

        # Base Represenations for all core states and candidate actions
        self.T_repr = torch.FloatTensor([]).to("cpu")

        # MDP Tensors
        self.Ti = torch.tensor([]).type(torch.LongTensor).to("cuda") # (nn, aa, tt)  ||  Transiton Indexes
        self.Tp = torch.tensor([]).type(torch.float32).to("cuda") # (nn, aa, tt)  ||  Transition Probabilities
        self.R = torch.tensor([]).type(torch.float32).to("cuda") # (nn, aa, tt)  ||  Transition Rewards
        self.Tdist = torch.tensor([]).type(torch.float32).to("cuda") # (nn, aa, tt)  ||  Distances for s,a,s' approximations
        self.P = torch.tensor([]).type(torch.float32).to("cuda") # (nn, aa, tt)  ||  Penalites for s,a,s' approximations
        self.Q = torch.tensor([]).type(torch.float32).to("cuda") # (nn, aa)  ||  Q values for each state tran_type pair
        self.V = torch.tensor([]).type(torch.float32).to("cuda") # (nn,)  ||  Value Vector
        self.C = torch.tensor([]).type(torch.float32).to("cuda") # (nn,)  ||  Cost of optimal policy for each state.
        self.Pi = torch.tensor([]).type(torch.LongTensor).to("cuda") # (nn,)  ||  Policy Vector

        # MDP Solve helper variables
        self.gamma = torch.FloatTensor([0.99]).to(device)


    def init_transitions(self, transitions: DACTransitionBatch, replace_at_indices = None, verbose = False):
        # all_states = torch.tensor([t[0] for t in  transitions]).type(torch.FloatTensor)
        # all_actions = torch.tensor([t[1] for t in  transitions]).type(torch.FloatTensor)
        # self.transitions = transitions
        batch_next_states = transitions.next_states.clone().detach().type(torch.FloatTensor)
        batch_rewards = transitions.rewards.clone().detach().type(torch.FloatTensor)
        batch_terminals = transitions.terminals.clone().detach().type(torch.LongTensor)
        batch_terminal_indices = batch_terminals.nonzero().type(torch.LongTensor).reshape(-1)
        batch_next_states[batch_terminal_indices] = 999999

        
        print("batch_next_states.shape", batch_next_states.shape )
        print("replace indices : ", replace_at_indices is not None, batch_next_states.shape )
        
        # update next states
        bb = len(batch_next_states)
        nn = (len(self.S) + bb) if replace_at_indices is None else len(self.S) 
        aa, tt = self.dac_constants.n_tran_types, self.dac_constants.n_tran_targets

        ####################### Append Dataset Caches and base represenations##################
        if replace_at_indices is None:
            self.D_rewards = torch.concat([self.D_rewards, batch_rewards.to("cuda")])
            self.D_terminals = torch.concat([self.D_terminals, batch_terminals.to("cuda")])
            self.D_repr = torch.concat([self.D_repr, torch.zeros((bb, self.dac_constants.sa_repr_dim)).type(torch.float32).to("cuda")], dim = 0)
            self.T_repr = torch.concat([self.T_repr, torch.zeros((bb, aa, self.dac_constants.sa_repr_dim)).type(torch.float32).to("cpu")], dim = 0)
        else:
            for T in [self.D_repr, self.T_repr]:
                T[replace_at_indices] = 0
            self.D_rewards[replace_at_indices] = batch_rewards.to("cuda")
            self.D_terminals[replace_at_indices] = batch_terminals.to("cuda")
                
        self.D_terminal_indices = self.D_terminals.nonzero().type(torch.LongTensor).reshape(-1)
        assert self.D_rewards.shape == (nn,), f"D_rewards shape : {self.D_rewards.shape}, {nn}"
        assert self.D_repr.shape == (nn, self.dac_constants.sa_repr_dim)
        assert self.T_repr.shape == (nn, aa, self.dac_constants.sa_repr_dim)
        ###############################################################################################################

        ####################### Append Core States ##################
        if replace_at_indices is None:
            self.S = torch.concat([self.S, batch_next_states.to("cuda")], dim = 0)  # core states contains only target vectors.
        else:
            self.S[replace_at_indices] = batch_next_states.to("cuda")  # core states contains only target vectors.
            
        assert self.S.size(0) == nn
        ###############################################################################################################

        ####################### Append MDP Tensors ##################
        if replace_at_indices is None: 
            self.Tp = torch.concat([self.Tp, torch.zeros((bb, aa, tt)).type(torch.float32).to("cuda")], dim = 0)
            self.Ti = torch.concat([self.Ti, torch.zeros((bb, aa, tt)).type(torch.LongTensor).to("cuda")], dim = 0)
            self.R = torch.concat([self.R, torch.zeros((bb, aa, tt)).type(torch.float32).to("cuda")], dim = 0)
            self.Tdist = torch.concat([self.Tdist, torch.zeros((bb, aa, tt)).type(torch.float32).to("cuda")], dim = 0) # Distances for s,a,s' approximations
            self.P = torch.concat([self.P, torch.zeros((bb, aa, tt)).type(torch.float32).to("cuda")], dim = 0) # Penalites for s,a,s' approximations
            self.Q = torch.concat([self.Q, torch.zeros((bb, aa)).type(torch.float32).to("cuda")], dim = 0)
            self.V = torch.concat([self.V, torch.zeros((bb,)).type(torch.float32).to("cuda")], dim = 0)
            self.C = torch.concat([self.C, torch.zeros((bb,)).type(torch.float32).to("cuda")], dim = 0) # Cost of optimal policy for each state.
            self.Pi = torch.concat([self.Pi, torch.zeros((bb)).type(torch.LongTensor).to("cuda")], dim = 0)
        else:
            for T in [self.Tp, self.Ti, self.R, self.Tdist, self.P, self.Q, self.V, self.C, self.Pi]:
                T[replace_at_indices] = 0
                
        assert self.Tp.shape == (nn, aa, tt)
        assert self.Ti.shape == (nn, aa, tt)
        assert self.R.shape == (nn, aa, tt)
        assert self.Tdist.shape == (nn, aa, tt)
        assert self.P.shape == (nn, aa, tt)
        assert self.Q.shape == (nn, aa)
        assert self.V.shape == (nn,)
        assert self.C.shape == (nn,)
        assert self.Pi.shape == (nn,)
        ###############################################################################################################
        
        print(f"Instantiated DACMDP for transition Batch")
        print((nn, aa, tt))
                

    def set_transition_reprsentations(self, tran_repr_sets: torch.tensor, state_indices: torch.tensor) -> bool:
        """ each transtition representation represents a transtion type and is used to
        infer the target states.
        Returns:
            _type_: _description_
        """
        n, a, t = tran_repr_sets.shape
        assert a == self.dac_constants.n_tran_types and t == self.dac_constants.sa_repr_dim and n == len(state_indices)
        self.T_repr[state_indices] = tran_repr_sets.to("cpu")
        return True

    def set_dataset_representations(self, sa_reprs: torch.tensor, state_indices: torch.tensor) -> bool:
        n, r_dim = sa_reprs.shape
        assert len(state_indices) == n and r_dim == self.dac_constants.sa_repr_dim
        self.D_repr[state_indices] = sa_reprs.to(self.device)
        return True


    def update_tran_vectors(self, state_indices: torch.tensor) -> bool:
        # Filter state_indices for terminal indices 
        # we never update transitions for them.
        
        # Get KNN and calculate transition tensors
        T_repr_slice = self.T_repr[state_indices]
        nn, aa, s_dim = T_repr_slice.shape
        all_sa_reprs = T_repr_slice.view((-1, self.dac_constants.sa_repr_dim)).cuda()
        nn_indices_flat, nn_values_flat = self.batch_calc_knn_ret_flat_engine(all_sa_reprs, self.D_repr, k=self.dac_constants.n_tran_targets)
        knn_idx_tensor = nn_indices_flat.view((nn, aa, self.dac_constants.n_tran_targets)).to(self.device)
        knn_values_tensor = nn_values_flat.view((nn, aa, self.dac_constants.n_tran_targets)).to(self.device)

        assert self.Ti[state_indices].shape == knn_idx_tensor.shape
        assert self.Tp[state_indices].shape == knn_values_tensor.shape

        self.Ti[state_indices] = knn_idx_tensor
        self.Tdist[state_indices] = knn_values_tensor
        self.R[state_indices] = self.D_rewards[knn_idx_tensor.view(-1)].reshape(knn_idx_tensor.shape)
        if self.dac_constants.penalty_type == "linear":
            # self.P[state_indices] = self.penalty_beta * torch.exp(self.Tdist[state_indices])
            self.P[state_indices] = self.dac_constants.penalty_beta * self.Tdist[state_indices]
            self.Tp[state_indices] = torch.nn.Softmax(dim = 2)(1/(knn_values_tensor+0.0001))
        elif self.dac_constants.penalty_type == "exponential":
            self.P[state_indices] = self.dac_constants.penalty_beta * self.Tdist[state_indices]
            self.Tp[state_indices] = torch.nn.Softmax(dim = 2)(torch.log(1/(knn_values_tensor+0.0001)))
        else: 
            assert False, f"logic for penalty type {self.dac_constants.penalty_type} not defined"
        
        # Reset for terminal transitions , no need to update
        self.Ti[self.D_terminal_indices] = 0
        self.Tp[self.D_terminal_indices] = 0
        self.R[self.D_terminal_indices] = 0
        self.P[self.D_terminal_indices] = 0

        return True

    def single_bellman_backup_computation(self):
        self.curr_error, self.Q, self.Pi, self.V = DACMDP_CORE.bellman_backup_operator(self.Ti, self.Tp, self.R, self.P, self.Q, self.V, self.gamma)

    def single_dac_bellman_backup_computation(self):
        self.curr_error, self.Q, self.Pi, self.V, self.C = DACMDP_CORE.dac_bellman_backup_operator(self.Ti, self.Tp, self.R, self.P, self.Q, self.V, self.C, self.gamma)

    def single_safe_bellman_backup_computation(self):
        self.curr_error, self.Q, self.Pi, self.V, self.C = DACMDP_CORE.dac_safe_bellman_backup_operator(self.Ti, self.Tp, self.R, self.P, self.Q, self.V, self.C, self.gamma)

    def solve(self, max_n_backups=500, gamma=0.99, epsilon=0.001, penalty_beta = 1, operator = "simple_backup", reset_values = False, verbose=False, bellman_backup_batch_size=250) -> None:
        if reset_values:
            self.reset_value_vectors()
        if self.dac_constants.penalty_beta != penalty_beta:
            self.reset_value_vectors()
            self.update_penalty_beta(penalty_beta)

        operator_map = {"simple_backup":self.single_bellman_backup_computation, 
                        "dac_backup":self.single_dac_bellman_backup_computation,
                        "safe_backup":self.single_safe_bellman_backup_computation}

        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        v_iter = tqdm(range(max_n_backups)) if verbose else range(max_n_backups)

        for i in v_iter:
            operator_map[operator]()
            if i % bellman_backup_batch_size == 0:
                print(i, self.curr_error.cpu())
                if self.curr_error.cpu() < epsilon:
                    break
        print(f"Solved MDP in {i} Backups")



    
    # Extenstion Functions 
    def reset_value_vectors(self):
        nn,aa,tt = self.Ti.shape
        self.Q = torch.zeros((nn, aa)).type(torch.float32).to(device=self.device)
        self.V = torch.zeros((nn,)).type(torch.float32).to(device=self.device)
        self.C = torch.zeros((nn,)).type(torch.float32).to(device=self.device) # Cost of optimal policy for each state.
        self.Pi = torch.zeros((nn)).type(torch.LongTensor).to(device=self.device)

    def update_penalty_beta(self,penalty_beta):
        self.penalty_beta = penalty_beta
        self.P = penalty_beta * self.Tdist
    
    def update_dist_normalization(self,exp_dist = False):
        if exp_dist:
            self.P = torch.nn.Softmax(dim = 2)(1/(self.Tdist+0.0001))
        else:
            self.P = torch.nn.Softmax(dim = 2)(torch.log(1/(self.Tdist+0.0001)))

    def calc_dynamics_prob_using_true_softmax(self):
        self.Tp = torch.nn.Softmax(dim=2)(1/(self.Tdist+0.0001)).to(self.device)
        


class DACMDP(DACMDP_CORE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    ## Logging Functions
    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        for t_name in ["D_repr", "T_repr", "Tp", "Ti", "R", "Tdist", "P", "Q", "V", "C", "Pi", "S"]:
            s = f"torch.save(self.{t_name}, '{save_folder}/{t_name}.torch')"
            exec(s)

    def load(self, save_folder):
        for t_name in ["D_repr", "T_repr", "Tp", "Ti", "R", "Tdist", "P", "Q", "V", "C", "Pi", "S"]:
            exec(f"self.{t_name} = torch.load('{save_folder}/{t_name}.torch')")

    @property
    def mdp_distributions(self):
        s_count = len(self.S)
        all_distr = {"Transition Probabilty Distribution": self.Tp.reshape(-1).to("cpu").numpy(),
             "Raw Reward Distribution": self.R.reshape(-1).to("cpu").numpy(),
             "Penalty Distribution": self.P.reshape(-1).to("cpu").numpy(),
             "DAC Reward Distribution": (self.R.reshape(-1) - self.P.reshape(-1)).to("cpu").numpy(),
             "Cost [cuml penalties] Distribution": self.C.reshape(-1).to("cpu").numpy(),
             "Value [cuml rewards] Distribution": self.V.reshape(-1).to("cpu").numpy(),
             }
        return all_distr 