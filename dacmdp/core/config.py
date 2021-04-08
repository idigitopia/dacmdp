import os
import hashlib
import gym
from collections import defaultdict

ACTION_SCALE = defaultdict(lambda: defaultdict(lambda: 1))


class BaseConfig(object):
    def __init__(self, args, dynamics_hp_group, agent_hp_group):
        self.args = args
        dyn_hp_hash = hashlib.sha224(bytes(''.join(sorted([str(vars(args)[hp.dest])
                                                           for hp in sorted(dynamics_hp_group._group_actions,
                                                                            key=lambda x: x.dest)])),
                                           'ascii')).hexdigest()
        agent_hp_hash = hashlib.sha224(bytes(''.join(sorted([str(vars(args)[hp.dest])
                                                             for hp in sorted(agent_hp_group._group_actions,
                                                                              key=lambda x: x.dest)])),
                                             'ascii')).hexdigest()
        self.dynamics_exp_path = os.path.join(args.result_dir, args.case, args.env_name, dyn_hp_hash)
        self.agent_exp_path = os.path.join(self.dynamics_exp_path, agent_hp_hash)

        # store env attributes
        env = self.new_game()
        self.observation_size = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.action_size = env.action_space.shape[0]
        self.sample_random_action = lambda: env.action_space.sample()
        self.symbolic_env = True

    def get_uniform_dynamics_network(self):
        return EnsembleDynamicsNetwork(n=self.args.num_ensemble,
                                       obs_size=self.observation_size,
                                       state_size=self.args.state_size,
                                       hidden_size=self.args.hidden_size,
                                       action_size=self.action_size,
                                       symbolic=self.symbolic_env)

    def get_uniform_agent_network(self):
        return AgentNetwork(state_size=self.args.state_size,
                            hidden_size=self.args.hidden_size,
                            action_size=self.action_size,
                            sample_random_action_fn=self.sample_random_action,
                            action_scale=self.action_scale)

    def get_uniform_ppo_agent_network(self):
        return PPOActorCriticNetwork(num_inputs=self.observation_size,
                                     num_actions=self.action_size,
                                     hidden_dim=64,
                                     action_space=self.action_space,
                                     action_std=0.5,
                                     action_scale=self.action_scale)

    def new_game(self):
        return gym.make('d4rl:' + self.args.env_name)

    @property
    def action_scale(self):
        return ACTION_SCALE[self.args.case][self.args.env_name]

    def get_hparams(self):
        hparams = {k: v for k, v in vars(self.args).items() if v is not None}
        for k, v in self.__dict__.items():
            if 'path' not in k and 'args' not in k and 'sample_random_action' not in k and (v is not None):
                hparams[k] = v
        return hparams

    @property
    def dynamics_model_path(self):
        return os.path.join(self.dynamics_exp_path, 'dynamics_model.p')

    @property
    def dynamics_checkpoint_path(self):
        return os.path.join(self.dynamics_exp_path, 'dynamics_checkpoint.p')

    @property
    def agent_model_path(self):
        return os.path.join(self.agent_exp_path, 'agent_model.p')

    @property
    def agent_checkpoint_path(self):
        return os.path.join(self.agent_exp_path, 'agent_checkpoint.p')

    @property
    def dynamics_logs_path(self):
        return os.path.join(self.dynamics_exp_path, 'logs')

    @property
    def agent_logs_path(self):
        return os.path.join(self.agent_exp_path, 'logs')