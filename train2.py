from nes_py.wrappers import JoypadSpace
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnCnnPolicy, CnnPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.callbacks import CallbackList, EvalCallback
from callbacks import ProgressBarManager
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
    env = JoypadSpace(env, RIGHT_ONLY)

    model = DQN(LnCnnPolicy, 
        env, 
        verbose=1, 
        exploration_fraction=0.9, 
        prioritized_replay=True, 
        train_freq=10, 
        learning_starts=750,
        tensorboard_log="./mario_tensorboard/"
    )

    with ProgressBarManager(1500) as callback:
        model.learn(total_timesteps=1500, log_interval=1, callback=callback)
    model.save("mario_mdl")

if __name__ == '__main__':
    main()