from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv
from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks import ProgressBarManager
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print ("Setting up environment...")
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, RIGHT_ONLY)
env = WarpFrame(env)
env = FrameStack(env, n_frames=4)
# env = MaxAndSkipEnv(env)

# from matplotlib import pyplot as plt
# import cv2

# obs = env.reset()
# while True:
#     obs, rewards, dones, info = env.step(env.action_space.sample())
#     print(obs._frames)
#     cv2.imshow("frames", obs._frames[0])
#     cv2.waitKey()
#     # img = Image.fromarray(obs._frames, 'RGB')
#     # img.show()
#     # print("reward:", rewards)
#     print("timestep:", info['timestep'])


# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=2500, save_path='./logs/',
                                         name_prefix='model')

eval_callback = EvalCallback(env,
                             best_model_save_path='./logs/',
                             log_path='./logs/',
                             eval_freq=5000,
                             deterministic=True,
                             render=False)

print("Compiling model...")
model = DQN(CnnPolicy,
            env,
            verbose=1,
            learning_starts=10000,
            learning_rate=1e-4,
            exploration_final_eps=0.01,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            tensorboard_log="./mario_tensorboard/"
        )

print("Training starting...")
steps = 20000
with ProgressBarManager(steps) as progress_callback:
    model.learn(total_timesteps=steps,
                callback=[progress_callback, eval_callback, checkpoint_callback],
                tb_log_name="dqn-framestack")

print("Done! Saving model...")
model.save("dqn")