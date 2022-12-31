import numpy as np


def initialize_env(env, state, env_name):

    if env_name in [
        "CartPole-cont-v1", "CartPole-cont-v0", "CartPole-cont-noisy-v1", "CartPole-cont-noisy-v0"]:
        init_fxn = init_fxn_for_cartpole_cont
    elif "maze2d" in env_name:
        init_fxn = init_fxn_for_maze2d
    elif "antmaze" in env_name:
        init_fxn = init_fxn_for_antmaze
    elif "halfcheetah" in env_name:
        init_fxn = init_fxn_for_halfcheetah
    elif "walker2d" in env_name:
        init_fxn = init_fxn_for_walker2d
    elif "hopper" in env_name:
        init_fxn = init_fxn_for_hopper
    else:
        assert False, f"Oracle Function Not Defined for env {env_name} yet"

    return init_fxn(env, state)

def init_fxn_for_cartpole_cont(env, s):
    env.unwrapped.state = s
    return env

def init_fxn_for_maze2d(env, s):
    env.unwrapped.set_state(s[:2], s[2:])
    return env

def init_fxn_for_halfcheetah(env, s):
    env.unwrapped.set_state(np.array([0] + s[:8].tolist()), s[8:])
    return env

def init_fxn_for_antmaze(env, s):
    env.unwrapped.set_state(s[:15], s[15:])
    return env

def init_fxn_for_walker2d(env, s):
    env.unwrapped.set_state(np.array([0] + s[:8].tolist()), s[8:])
    return env

def init_fxn_for_hopper(env, s):
    env.unwrapped.set_state(np.array([0] + s[:5].tolist()), s[5:])
    return env