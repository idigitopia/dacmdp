from gym.envs.registration import register
# import d4rl
# d4rl.infos.REF_MIN_SCORE.update({"CartPole-cont-v1":15})
# d4rl.infos.REF_MAX_SCORE.update({"CartPole-cont-v1":500})
# d4rl.infos.REF_MIN_SCORE.update({"CartPole-cont-noisy-v1":15})
# d4rl.infos.REF_MAX_SCORE.update({"CartPole-cont-noisy-v1":500})

# d4rl.infos.REF_MIN_SCORE.update({"CartPole-cont-v0":15})
# d4rl.infos.REF_MAX_SCORE.update({"CartPole-cont-v0":200})
# d4rl.infos.REF_MIN_SCORE.update({"CartPole-cont-noisy-v0":15})
# d4rl.infos.REF_MAX_SCORE.update({"CartPole-cont-noisy-v0":200})

register(
    id='CartPole-cont-v0',
    entry_point='dacmdp.envs.cont_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-cont-v1',
    entry_point='dacmdp.envs.cont_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CartPole-cont-noisy-v0',
    entry_point='dacmdp.envs.cont_noisy_cartpole:ContinuousCartPoleNoisyEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-cont-noisy-v1',
    entry_point='dacmdp.envs.cont_noisy_cartpole:ContinuousCartPoleNoisyEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)