from nes_py.wrappers import JoypadSpace
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
import retro
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, RIGHT_ONLY)
env = WarpFrame(env)
env = FrameStack(env, n_frames=4)
env = EpisodicLifeEnv(env)



env_render = gym_super_mario_bros.make('SuperMarioBros-v0')
env_render = JoypadSpace(env_render, RIGHT_ONLY)

model = DQN.load("best_model_1")

obs = env.reset()
env_render.reset()

cr = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env_render.step(action)
    cr += rewards
    print("reward:{}\t\t".format(cr), end='\r')
    env_render.render()
    if (done):
        print("finished an episode with total reward:", cr)
        cr = 0
        break
print("Done")
