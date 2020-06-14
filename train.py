import os
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import CnnPolicy, LnCnnPolicy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks import ProgressBarManager

# CONSTANTS
TIME_STEPS = 10000
# Data preprocessing & other Atari wrappers
FRAME_STACK = 4  # Nature uses 4
FRAME_SKIP = 8  # Nature uses 4
# Neural network
BATCH_SIZE = 96
LEARNING_RATE = 1e-4
EXPLORATION_FRACT = 0.1  # Nature uses 0.1
P_REPLAY_ALPHA = 0.6
TRAINING_FREQ = 4
TARGET_UPDATE_FREQ = 10000

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


def run(model_name):
    print("Setting up environment...")

    # Create log dir
    log_dir = "./monitor_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Environment setup
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = EpisodicLifeEnv(env)

    # Preprocessing
    env = WarpFrame(env)
    env = FrameStack(env, n_frames=FRAME_STACK)

    # Evaluate every kth frame and repeat action
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/',
                                             name_prefix=model_name)

    eval_callback = EvalCallback(env,
                                 best_model_save_path='./logs/',
                                 log_path='./logs/',
                                 eval_freq=1000,
                                 deterministic=False,
                                 render=False)

    print("Compiling model...")

    try:
        model = DQN.load("models/{}".format(model_name))
    except:
        pass

        # The probability of greedy actions from DQN increases from 0 to 0.95 linearly during the first 75% steps
    model = DQN(LnCnnPolicy,
                env,
                gamma=0.99,
                learning_rate=LEARNING_RATE,  # 1e-4 for QMap paper
                buffer_size=500000,  # 500,000 for QMap paper
                exploration_fraction=EXPLORATION_FRACT,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.1,
                train_freq=4,  # 4 for QMap paper
                batch_size=BATCH_SIZE,
                double_q=True,
                # QMap paper: training starts after 1000 steps and the networks are used after 2000 steps
                learning_starts=1000,
                target_network_update_freq=1000,  # 1000 for QMap paper
                prioritized_replay=True,
                prioritized_replay_alpha=0.6,
                prioritized_replay_beta0=0.4,
                prioritized_replay_beta_iters=None,
                prioritized_replay_eps=1e-6,
                param_noise=False,
                n_cpu_tf_sess=None,
                verbose=1,
                tensorboard_log="./mario_tensorboard/",
                policy_kwargs=None,
                full_tensorboard_log=False,
                )

    print("Training starting...")
    with ProgressBarManager(TIME_STEPS) as progress_callback:
        model.learn(total_timesteps=TIME_STEPS,
                    # , eval_callback, checkpoint_callback],
                    callback=[progress_callback],
                    tb_log_name=model_name)

    print("Done! Saving model...")
    model.save("models/{}".format(model_name))


if __name__ == "__main__":
    run("dqn")
