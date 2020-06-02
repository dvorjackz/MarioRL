from nes_py.wrappers import JoypadSpace
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.callbacks import CallbackList, EvalCallback
from callbacks import ProgressBarManager
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print ("Setting up environment...")
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

eval_callback = EvalCallback(env,
                             best_model_save_path='./logs/',
                             log_path='./logs/',
                             eval_freq=500,
                             deterministic=True,
                             render=False)

print("Compiling model...")
model = DQN(CnnPolicy,
            env,
            verbose=1,
            learning_starts=750,
            learning_rate=1e-4,
            exploration_final_eps=0.01,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            double_q=False,
            tensorboard_log="./mario_tensorboard/"
        )

print("Training starting...")
steps = 1500
with ProgressBarManager(steps) as progress_callback:
    model.learn(total_timesteps=steps,
                callback=[progress_callback],
                tb_log_name="run")

print("Done! Saving model...")
model.save("dqn")