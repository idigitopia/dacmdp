from sklearn.neighbors import KDTree as RawKDTree
import gym
import random
import torch
import numpy as np
from . import utils_misc as utils
from .utils_knn import THelper 
from sklearn.cluster import KMeans

import math

def _effective_batch_size(lookup_dim):
    nn, dd = lookup_dim
    batch_size = min(nn,int(1000*(64/dd)*(1000000/nn)))
    return batch_size

class CustomActionSpace():

    def __init__(self, action_list):
        self.action_list = np.array(action_list)
        self.shape = action_list[0].shape

    def sample(self,):
        return random.choice(self.action_list)

    def __len__(self):
        return len(self.action_list)


def get_action_list_from_space(action_space):
    """
    Returns a list of actions for the input gym space.
    """
    if isinstance(action_space, gym.spaces.discrete.Discrete):
        return [(i,) for i in list(range(0, action_space.n))]
    if isinstance(action_space, CustomActionSpace):
        return action_space.action_list
    else:
        print("No parse logic defined for Action Space" + str(type(action_space)))
        print("Warning Returning: None")
        return [None]


def get_cardinality_of_action_space(action_space):
    if isinstance(action_space, gym.spaces.discrete.Discrete):
        return action_space.n
    elif isinstance(action_space, CustomActionSpace):
        return len(action_space.action_list)
    else:
        print("Warning: No cardinality logic defined for Action Space" + str(type(action_space)))
        print("Warnint: Returning None")
        return None


def sample_from_action_space(action_space):
    if isinstance(action_space, gym.spaces.discrete.Discrete):
        return [action_space.sample()]
    else:
        return action_space.sample()


class BaseActionModel:
    """
    Designed to be inherited by all action models. 
    random_action methods designed to work with all action models
    """

    def __init__(self, action_space, n_actions=None):

        # if gym.spaces.discrete.
        self.action_space = action_space

        self.n_actions = n_actions or get_cardinality_of_action_space(action_space)
        assert self.n_actions, "n_actions must be provided for action_space: " + str(type(action_space))

    def __repr__(self):
        return "BaseActionModel"

    def cand_actions_for_states(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def cand_actions_for_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.cand_actions_for_states(state.unsqueeze(0))[0]

    def random_action(self):
        return torch.FloatTensor(sample_from_action_space(self.action_space))

    def random_actions(self):
        return torch.stack([self.random_action() for _ in range(self.n_actions)])


class RandomActionModel(BaseActionModel):
    """
    Designed to be inherited by all action models. 
    random_action methods designed to work with all action models
    """

    def __init__(self, action_space, n_actions=None):
        super().__init__(action_space, n_actions)

    def cand_actions_for_state(self, state) -> torch.Tensor:
        return torch.FloatTensor(self.random_actions())

    def cand_actions_for_states(self, states) -> torch.Tensor:
        return torch.stack([self.random_actions() for _ in states])

    def __repr__(self):
        return "RandomActionModel"


class DiscreteActionModel(BaseActionModel):

    def __init__(self, action_space, n_actions=None):
        """ Designed to work with DiscreteAction Space and Custom Action Space.

        Args:
            action_space ([type]): [description]
            n_actions ([type], optional): [description]. Defaults to None.
        """
        super().__init__(action_space, n_actions)

        self.action_list = get_action_list_from_space(action_space)

    def cand_actions_for_state(self, state) -> torch.Tensor:
        return torch.FloatTensor(self.action_list)

    def cand_actions_for_states(self, states) -> torch.Tensor:
        return torch.stack([self.cand_actions_for_state(s) for s in states])

    def __repr__(self):
        return "DiscreteActionModel"


class NNActionModel(BaseActionModel):

    def __init__(self, action_space, n_actions, data_buffer, nn_engine = "kd_tree", projection_fxn = None):

        # will only work for feature representation of states, not images.
        # one can also do a random projection step here if images are passed.
        super().__init__(action_space, n_actions)

        # make a list of all seen states .
        
        # make a list of all seen states .
        self.projection_fxn = projection_fxn or (lambda s:s)
        self.latent_S = utils.v_map(fxn=projection_fxn,
                                     iterable=torch.FloatTensor(data_buffer.all_states),
                                     batch_size=256,
                                     reduce_fxn=lambda x: torch.cat(x, dim=0),
                                     label="Caculating State Representations").cuda()
        self.action_store = torch.FloatTensor(data_buffer.all_actions)
        self.n_actions = n_actions

        # create a kd tree.
        if nn_engine == "kd_tree":
            self.s_kDTree = RawKDTree(self.latent_S.cpu().numpy())
            self.batch_query_knn_s_idxs = lambda s_batch, k: self.s_kDTree.query(s_batch, k=k)[1]
        elif nn_engine == "torch_jit":
            self.batch_query_knn_s_idxs = lambda s_batch, k: THelper.batch_calc_knn_jit(s_batch.cuda(), self.latent_S, k=k)[0].cpu()
        elif nn_engine == "torch_pykeops":
            self.batch_query_knn_s_idxs = lambda s_batch, k: THelper.batch_calc_knn_pykeops(s_batch.cuda(), self.latent_S, k=k)[0].cpu()
        else:
            print(f"nn_engine {nn_engine} not defined. Switching to default. torch_jit")
            self.batch_query_knn_s_idxs = lambda s_batch, k: THelper.batch_calc_knn_jit(s_batch.cuda(), self.latent_S, k=k)[0].cpu()

    def cand_actions_for_state(self, state: torch.Tensor) -> torch.Tensor:
        # query for nearest neighbors.
        knn_idxs = self.batch_query_knn_s_idxs(self.projection_fxn(state.unsqueeze(0)).to("cuda"), k=self.n_actions)[0]
        return self.action_store[knn_idxs.cpu()]

    def cand_actions_for_states(self, states: torch.Tensor) -> torch.Tensor:
        # query for nearest neighbors.
        knn_idxs_batch = self.batch_query_knn_s_idxs(self.projection_fxn(states).to("cuda"), k=self.n_actions)
        return self.action_store[knn_idxs_batch.cpu()]

    def __repr__(self):
        return "NNActionModel"


from .utils_knn import KMeans as KMeans_plykeops
from .utils_knn import KMeans_cosine as KMeans_cosine_plykeops

class GlobalClusterActionModel(BaseActionModel):

    def __init__(self, action_space, n_actions, data_buffer, cosine_dist = False, engine = "py_keops"):
        # will only work for feature representation of states, not images.
        # one can also do a random projection step here if images are passed.
        super().__init__(action_space, n_actions)

        self.cosine_dist = cosine_dist
        
        action_array = torch.FloatTensor(data_buffer.all_actions).cuda()

        if engine == "sklearn":
            from sklearn.cluster import KMeans
            # Create KMeans instance
            kmeans = KMeans(n_clusters=n_actions)
            kmeans.fit(action_array.cpu().numpy()) 
            self.actions = torch.FloatTensor(kmeans.cluster_centers_).cpu()
            # Fit model
        else:
            if self.cosine_dist:
                cl,c = KMeans_cosine_plykeops(action_array, K=n_actions, Niter=50)
            else:
                cl,c = KMeans_plykeops(action_array, K=n_actions, Niter=50)

                self.actions = c.cpu()
        self.n_actions = n_actions

    def cand_actions_for_states(self, states) -> np.ndarray:
        return self.actions.repeat(len(states), 1, 1)

    def __repr__(self):
        return "GlobalClusterActionModel"
    

class EnsembleActionModel(BaseActionModel):

    def __init__(self, action_space, ensemble_action_models):

        total_n_actions = sum([m.n_actions for m in ensemble_action_models])
        super().__init__(action_space, total_n_actions)

        self.ensemble_action_models = ensemble_action_models

    # def join_cand_actions_for_state(self, ensemble_of_actions):
    #     return [a for actions in ensemble_of_actions for a in actions]

    def cand_actions_for_state(self, state):
        assert False, "Not Implemented Error"
        # ensemble_of_actions = [m.cand_actions_for_state(state) for m in self.ensemble_action_models]
        # return self.join_cand_actions_for_state(ensemble_of_actions)

    def cand_actions_for_states(self, states):
        return torch.cat([m.cand_actions_for_states(states) for m in self.ensemble_action_models], dim = 1)


class ActionModelStore(object):
    """
    Holds logic for easy fetching of sa_repr models.
    """

    def __init__(self, action_out_spec, data_buffer):
        """
        Specifies loading/initialize mechanism for easy fetch of sa repr models. 
        """
        self.action_space, self.n_actions = action_out_spec
        self.data_buffer = data_buffer

    def fetch(self, repr_model_name):
        # one can set or unset some variables here , fetch can take multiple arguments
        action_model = getattr(self, repr_model_name)
        return action_model

    @property
    def BaseActionModel(self):
        action_model = BaseActionModel(action_space=self.action_space,
                                       n_actions=self.n_actions)
        return action_model

    @property
    def RandomActionModel(self):
        action_model = RandomActionModel(action_space=self.action_space,
                                         n_actions=self.n_actions)
        return action_model

    @property
    def DiscreteActionModel(self):
        action_model = DiscreteActionModel(action_space=self.action_space,
                                           n_actions=self.n_actions)
        return action_model

    @property
    def NNActionModel(self):
        action_model = NNActionModel(action_space=self.action_space,
                                     n_actions=self.n_actions,
                                     data_buffer=self.data_buffer)
        return action_model