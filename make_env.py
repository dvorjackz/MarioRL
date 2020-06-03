import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines.common import set_global_seeds
from baselines.common.atari_wrappers import FrameStack
import retro

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = retro.RetroEnv(game='SuperMarioBros-Nes',use_restricted_actions=retro.Actions.DISCRETE)
        # env = FrameStack(env, k=4)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init