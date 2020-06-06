from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.atari_wrappers import WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from baselines.common.atari_wrappers import FrameStack
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, RIGHT_ONLY)
env = WarpFrame(env)
env = EpisodicLifeEnv(env)
env = FrameStack(env, 4)

model = DQN.load("best_model")
model.set_env(env)
np.savez("mario_expert", generate_expert_traj(model, None, env, n_timesteps=int(10000), n_episodes=10))
