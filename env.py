from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv

def get_env():
	env = gym_super_mario_bros.make('SuperMarioBros-v0')
	env = JoypadSpace(env, RIGHT_ONLY)
	env = MaxAndSkipEnv(env, 4)
	env = EpisodicLifeEnv(env)
	env = WarpFrame(env)
	env = FrameStack(env, 4)
	return env