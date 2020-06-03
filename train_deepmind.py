from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines.common.callbacks import CallbackList, EvalCallback
from callbacks import ProgressBarManager
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print ("Setting up environment...")
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, RIGHT_ONLY)
env = FrameStack(env, n_frames=4)

eval_callback = EvalCallback(env,
                             best_model_save_path='./logs/',
                             log_path='./logs/',
                             eval_freq=1000,
                             deterministic=True,
                             render=False)

print("Compiling model...")
model = DQN(LnCnnPolicy,
            env,
            verbose=1,
            learning_starts=1000,
            learning_rate=1e-4,
            exploration_final_eps=0.01,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
			train_freq=10,
            tensorboard_log="./mario_tensorboard/"
        )

print("Training starting...")
steps = 1500
with ProgressBarManager(steps) as progress_callback:
    model.learn(total_timesteps=steps,
                callback=[progress_callback, eval_callback],
                tb_log_name="altered_reward")

print("Done! Saving model...")
model.save("dqn")