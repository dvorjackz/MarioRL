from nes_py.wrappers import JoypadSpace
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv
import retro
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, RIGHT_ONLY)
env = WarpFrame(env)
# env = MaxAndSkipEnv(env)
env = FrameStack(env, n_frames=4)

model = DQN.load("best_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("reward:", rewards)
    env.render()
