from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks import ProgressBarManager
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test_env(env, frame_by_frame=False):
    obs = env.reset()
    while True:
        obs, rewards, dones, info = env.step(env.action_space.sample())
        print(obs._frames)
        if (frame_by_frame):
            cv2.imshow("frames", obs._frames[0])
            cv2.waitKey()
        else:
            env.render()
        print("reward:", rewards)
        print("timestep:", info['timestep'])


def run(run_name):
    print("Setting up environment...")
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = WarpFrame(env)
    env = FrameStack(env, n_frames=4)
    env = EpisodicLifeEnv(env)
    # env = MaxAndSkipEnv(env)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/',
                                             name_prefix=run_name)

    eval_callback = EvalCallback(env,
                                 best_model_save_path='./logs/',
                                 log_path='./logs/',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False)

    print("Compiling model...")
    steps = 5000

    model = DQN(CnnPolicy,
                env,
                verbose=1,
                learning_starts=2500,
                learning_rate=1e-4,
                exploration_final_eps=0.01,
                prioritized_replay=True,
                prioritized_replay_alpha=0.6,
                train_freq=4,
                tensorboard_log="./mario_tensorboard/"
                )

    print("Training starting...")
    with ProgressBarManager(steps) as progress_callback:
        model.learn(total_timesteps=steps,
                    # , eval_callback, checkpoint_callback],
                    callback=[progress_callback],
                    tb_log_name=run_name)

    print("Done! Saving model...")
    model.save("models/{}".format(run_name))


if __name__ == "__main__":
    run("dqn")
