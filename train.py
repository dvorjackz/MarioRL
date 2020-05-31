from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from baselines import deepq

from stable_baselines import A2C

from stable_baselines.deepq.policies import LnMlpPolicy, LnCnnPolicy
from stable_baselines import DQN

from stable_baselines.common.callbacks import BaseCallback
from tqdm.auto import tqdm

import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

def main():
    # Create log dir
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = Monitor(env, log_dir)

    # model = DQN(LnCnnPolicy, env, verbose=1, exploration_fraction=0.5, prioritized_replay=True, train_freq=10, learning_starts=20000)
    model = DQN.load("mario_mdl", env=env, learning_starts=5000)
    with ProgressBarManager(5500) as callback:
        model.learn(total_timesteps=5500, log_interval=1, callback=callback)
    model.save("mario_mdl")

    results_plotter.plot_results([log_dir], 5500, results_plotter.X_TIMESTEPS, "Mario's rewards")
    plt.show()

if __name__ == '__main__':
    main()