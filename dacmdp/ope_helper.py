import pickle as pk 
import os

import json
import os
import pickle
import pprint

from absl import app
from absl import flags
import d4rl  # pylint:disable=unused-import
import gym
import numpy as np
from munch import Munch
from torch import nn

class OPE_Policy(nn.Module):
  """D4RL policy."""

  def __init__(self, policy_file):
    super(OPE_Policy, self).__init__()
    
    weights = pickle.load(open(policy_file,"rb"))
    self.fc0_w = weights['fc0/weight']
    self.fc0_b = weights['fc0/bias']
    self.fc1_w = weights['fc1/weight']
    self.fc1_b = weights['fc1/bias']
    self.fclast_w = weights['last_fc/weight']
    self.fclast_b = weights['last_fc/bias']
    self.fclast_w_logstd = weights['last_fc_log_std/weight']
    self.fclast_b_logstd = weights['last_fc_log_std/bias']
    relu = lambda x: np.maximum(x, 0)
    self.nonlinearity = np.tanh if weights['nonlinearity'] == 'tanh' else relu

    identity = lambda x: x
    self.output_transformation = np.tanh if weights[
        'output_distribution'] == 'tanh_gaussian' else identity

  def act(self, state, noise = 0):
    x = np.dot(self.fc0_w, state) + self.fc0_b
    x = self.nonlinearity(x)
    x = np.dot(self.fc1_w, x) + self.fc1_b
    x = self.nonlinearity(x)
    mean = np.dot(self.fclast_w, x) + self.fclast_b
    logstd = np.dot(self.fclast_w_logstd, x) + self.fclast_b_logstd

    action = self.output_transformation(mean + np.exp(logstd) * noise)
    return action, mean



class OPE_Helper:
    def __init__(self, plcy_dir="/nfs/hpc/share/frg-students/Policies/d4rl", 
                 base_dir="/nfs/guille/afern/users/shrestaa/new_projects/deep_ope", 
                 offline_dataset = "d4rl"):
        
        self.plcy_dir = plcy_dir
        self.base_dir = base_dir
        self.policy_db = json.load(open(os.path.join(self.base_dir, f"{offline_dataset}_policies.json"), "rb"))
        
    def get_policies(self, env_name):
        policies = Munch()
        
        for pid, policy_metadata in enumerate(self.policy_db):
            if any([env_name in t for t in  policy_metadata["task.task_names"]]):
                policies[f"{env_name}_{str(pid%11).zfill(2)}"] = self.get_policy(env_name, pid)
        return policies
    
    def get_policy_ids(self, env_name):
        policy_ids = []
        for i, policy_metadata in enumerate(self.policy_db):
            if any([env_name in t for t in  policy_metadata["task.task_names"]]):
                policy_ids.append((i,str(i%11).zfill(2)))
                
        return policy_ids
    
    def get_policy(self,env_name, policy_id = 80):
        policy = Munch()
        policy_metadata = self.policy_db[policy_id]
        policy.id = policy_id
        policy.update(policy_metadata)
        policy.model = OPE_Policy(os.path.join(self.plcy_dir, policy_metadata['policy_path']))
        
        return policy