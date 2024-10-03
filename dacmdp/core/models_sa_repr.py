from sklearn.neighbors import KDTree as RawKDTree
from collections import defaultdict
import torch
import numpy as np
from . import utils_misc as utils
import gym
from .utils_knn import THelper 



class StateActionRepr(object):
    """ Holds API structure for State action represetnations

    Args:
        object ([type]): [description]
    """

    def __init__(self):
        pass

    def encode_state_action_pair(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_state_action_pairs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class WeightedConcatRepr(object):
    """[summary]
    Concatenation State and Action in weighs each representations separately. 

    Args:
        object ([type]): [description]
    """

    def __init__(self, s_multiplyer, a_multiplyer):
        self.s_multiplyer = s_multiplyer
        self.a_multiplyer = a_multiplyer

    def encode_state_action_pair(self, state: torch.Tensor, action: torch.Tensor) -> torch.tensor:

        assert isinstance(state, torch.Tensor) and isinstance(action, torch.Tensor)

        # flatten states and actions
        state_flat, action_flat = state.reshape(-1), action.reshape(-1)
        sa_repr = torch.cat((self.s_multiplyer * state_flat, self.a_multiplyer * action_flat),
                            dim=0)
        return sa_repr

    def encode_state_action_pairs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.tensor:

        assert isinstance(states, torch.Tensor) and isinstance(actions, torch.Tensor)

        # flatten states and actions (along the batch. )
        states_flat, actions_flat = states.reshape(len(states), -1), actions.reshape(len(actions), -1)
        sa_reprs = torch.cat((self.s_multiplyer * states_flat, self.a_multiplyer * actions_flat),
                             dim=1)
        return sa_reprs


class DeltaPredictonRepr(object):
    """[summary]
    Concatenation State and Action in weighs each representations separately. 
    finds the sa_pair in the dataset with the shortest sa_repr distance

    Args:
        object ([type]): [description]
    """

    def __init__(self, s_multiplyer, a_multiplyer, buffer, batch_knn_engine = THelper.batch_calc_knn_jit):
        self.s_multiplyer = s_multiplyer
        self.a_multiplyer = a_multiplyer
        self.concat_repr_model = WeightedConcatRepr(s_multiplyer, a_multiplyer)

        self.state_store = torch.FloatTensor(buffer.all_states)
        self.action_store = torch.FloatTensor(buffer.all_actions)
        self.next_state_store = torch.tensor(buffer.all_next_states)

        self.sa_repr_store = torch.stack([self.concat_repr_model.encode_state_action_pair(
            s, a) for s, a in zip(self.state_store, self.action_store)])
        

        self.batch_knn_engine = batch_knn_engine
        self.latent_S = self.sa_repr_store.cuda()
        self.batch_query_knn_s_idxs = lambda s_batch, k : self.batch_knn_engine(s_batch.to("cuda"), self.latent_S, k=k)[0].cpu()

    def encode_state_action_pair(self, state: torch.Tensor, action: torch.Tensor) -> torch.tensor:

        assert isinstance(state, torch.Tensor) and isinstance(action, torch.Tensor)

        # flatten states and actions
        sa_repr = self.concat_repr_model.encode_state_action_pair(state, action)
        nn_idx = self.batch_query_knn_s_idxs(sa_repr, k = 1)[0]
        delta = self.next_state_store[nn_idx] - self.state_store[nn_idx]
        out = torch.FloatTensor(state) + torch.FloatTensor(delta)
        return out.type(torch.float32)

    def encode_state_action_pairs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.tensor:

        assert isinstance(states, torch.Tensor) and isinstance(actions, torch.Tensor)

        # flatten states and actions (along the batch. )
        sa_reprs = self.concat_repr_model.encode_state_action_pairs(states, actions)
        nn_idxs = self.batch_query_knn_s_idxs(sa_reprs, k = 2)[:,0].reshape(-1)
        deltas = self.next_state_store[nn_idxs] - self.state_store[nn_idxs]
        out = states + deltas
        return out.type(torch.float32)


class OracleDynamicsRepr(object):
    """[summary]
    Concatenation State and Action in weighs each representations separately. 
    finds the sa_pair in the dataset with the shortest sa_repr distance

    Args:
        object ([type]): [description]
    """

    def __init__(self, env_name, norm_fxn=None, denorm_fxn=None):
        self.env_name = env_name
        self.eval_env = gym.make(self.env_name)
        self.eval_env.reset()
        self.norm_fxn = norm_fxn or (lambda s: s)
        self.denorm_fxn = denorm_fxn or (lambda s: s)

        if self.env_name in [
            "CartPole-cont-v1", "CartPole-cont-v0", "CartPole-cont-noisy-v1", "CartPole-cont-noisy-v0"]:
            self.predict = self.predict_for_cartpole_cont
        elif "maze2d" in self.env_name:
            self.predict = self.predict_for_maze2d
        elif "antmaze" in self.env_name:
            self.predict = self.predict_for_antmaze
        elif "halfcheetah" in self.env_name:
            self.predict = self.predict_for_halfcheetah
        elif "walker2d" in self.env_name:
            self.predict = self.predict_for_walker2d
        elif "hopper" in self.env_name:
            self.predict = self.predict_for_hopper
        else:
            assert False, f"Oracle Function Not Defined for env {self.env_name} yet"

    def wrap_predict(func):
        def wrapper(self, s, a):
            s = self.denorm_fxn(np.array(s))
            # self.eval_env.reset()
            ns = func(self, s, a)
            return torch.FloatTensor(self.norm_fxn(ns))
        return wrapper

    @wrap_predict
    def predict_for_cartpole_cont(self, s, a):
        self.eval_env.unwrapped.state = s
        ns, r, tm,tc, info = self.eval_env.step(np.array(a))
        return ns

    @wrap_predict
    def predict_for_maze2d(self, s, a):
        self.eval_env.unwrapped.set_state(s[:2], s[2:])
        ns, r, tm,tc, info = self.eval_env.step(np.array(a))
        return ns

    @wrap_predict
    def predict_for_halfcheetah(self, s, a):
        self.eval_env.unwrapped.set_state(np.array([0] + s[:8].tolist()), s[8:])
        ns, r, tm,tc, info = self.eval_env.step(np.array(a))
        return ns

    @wrap_predict
    def predict_for_antmaze(self, s, a):
        self.eval_env.unwrapped.set_state(s[:15], s[15:])
        ns, r, tm,tc, info = self.eval_env.step(np.array(a))
        return ns

    @wrap_predict
    def predict_for_walker2d(self, s, a):
        self.eval_env.unwrapped.set_state(np.array([0] + s[:8].tolist()), s[8:])
        ns, r, tm,tc, info = self.eval_env.step(np.array(a))
        return ns

    @wrap_predict
    def predict_for_hopper(self, s, a):
        self.eval_env.unwrapped.set_state(np.array([0] + s[:5].tolist()), s[5:])
        ns, r, tm,tc, info = self.eval_env.step(np.array(a))
        return ns

    def encode_state_action_pair(self, state: torch.Tensor, action: torch.Tensor) -> torch.tensor:

        assert isinstance(state, torch.Tensor) and isinstance(action, torch.Tensor)
        return self.predict(state, action)

    def encode_state_action_pairs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.tensor:

        assert isinstance(states, torch.Tensor) and isinstance(actions, torch.Tensor)

        return torch.stack([self.predict(s, a) for s, a in zip(states, actions)]).to(torch.float32)


class torch_net_sa_repr(object):
    """_summary_
    sa_repr_model wrapper defined using the provided repr_net. 
    repr_net must have encode function that takes sates and actions as input. 
    """

    def __init__(self, repr_net):
        self.repr_net = repr_net
        self.device = next(self.repr_net.parameters()).device

    def encode_state_action_pair(self, state: torch.Tensor, action: torch.Tensor) -> torch.tensor:
        assert isinstance(state, torch.Tensor) and isinstance(action, torch.Tensor)
        state, action = state.to(self.device).unsqueeze(0), action.to(self.device).unsqueeze(0)
        sa_repr = self.repr_net.encode(state.float(), action.float())
        return sa_repr.detach().cpu()

    def encode_state_action_pairs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.tensor:
        assert isinstance(states, torch.Tensor) and isinstance(actions, torch.Tensor)
        states, actions = states.to(self.device), actions.to(self.device)
        sa_reprs = self.repr_net.encode(states.float(), actions.float())
        return sa_reprs.detach().cpu()


class ReprModelStore:
    """
    Holds logic for easy fetching of sa_repr models.
    """

    def __init__(self, config, data_buffer, repr_net=None):
        """
        Specifies loading/initialize mechanism for easy fetch of sa repr models. 
        """
        print("new")
        assert utils.has_attributes(config, ["reprModelArgs"])
        assert utils.has_attributes(config.reprModelArgs, ["s_multiplyer", "a_multiplyer"])

        self.s_multiplyer = config.reprModelArgs.s_multiplyer
        self.a_multiplyer = config.reprModelArgs.a_multiplyer
        self.data_buffer = data_buffer
        self.env_name = config.envArgs.env_name
        self.config = config

    def fetch(self, repr_model_name):
        # one can set or unset some variables here , fetch can take multiple arguments
        return getattr(self, repr_model_name)

    @property
    def WeightedConcatRepr(self):
        assert self.s_multiplyer is not None
        assert self.a_multiplyer is not None
        repr_model = WeightedConcatRepr(s_multiplyer=self.s_multiplyer,
                                        a_multiplyer=self.a_multiplyer)
        return repr_model

    @property
    def OracleDynamicsRepr(self):
        if self.data_buffer.norm_params.is_state_normalized:
            norm_fxn = self.data_buffer.query_normalized_state
            denorm_fxn = self.data_buffer.query_denormalized_state
            repr_model = OracleDynamicsRepr(self.env_name, norm_fxn, denorm_fxn)
        else:
            repr_model = OracleDynamicsRepr(self.env_name)

        if self.data_buffer.norm_params.is_action_normalized:
            assert False, "Action cannot be normalized for Oracle Dynamics"

        return repr_model

    @property
    def DeltaPredictonRepr(self):
        repr_model = DeltaPredictonRepr(s_multiplyer=self.s_multiplyer,
                                        a_multiplyer=self.a_multiplyer,
                                        buffer=self.data_buffer)
        return repr_model
