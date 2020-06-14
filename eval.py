from nes_py.wrappers import JoypadSpace
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import hyperparams as hp

env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = EpisodicLifeEnv(env)
env = WarpFrame(env)
env = FrameStack(env, n_frames=hp.FRAME_STACK)
env = MaxAndSkipEnv(env, skip=hp.FRAME_SKIP)

model = DQN.load("models/round3/best_model")

obs = env.reset()

# cr = 0
# while True:
#     action, _states = model.predict(obs, deterministic=False)
#     obs, rewards, done, info = env.step(action)
#     env.step(action)
#     cr += rewards
#     print("Reward: {}\t\t".format(cr), end='\r')
#     env.render()
#     if (done):
#         print("Finished an episode with total reward: ", cr)
#         cr = 0
#         break

print(evaluate_policy(model, env, n_eval_episodes=10, deterministic=False, render=True))

print("Done.")
