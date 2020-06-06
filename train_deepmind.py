from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy
from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks import ProgressBarManager
from env import get_env
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os
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

    print ("Setting up environment...")
    env = get_env()

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                            name_prefix=run_name)

    eval_callback = EvalCallback(env,
                                best_model_save_path='./logs/',
                                log_path='./logs/',
                                eval_freq=10000,
                                deterministic=True,
                                render=False)

    print("Compiling model...")
    steps = 500000

    model = DQN(CnnPolicy,
                env,
                verbose=1,
                learning_starts=2500,
                learning_rate=5e-5,
                exploration_final_eps=0.01,
                prioritized_replay=True,
                prioritized_replay_alpha=0.6,
                buffer_size=10000,
                train_freq=1,
                target_network_update_freq=1000,
                tensorboard_log="./mario_tensorboard/"
            )

    print("Training starting...")
    with ProgressBarManager(steps) as progress_callback:
        model.learn(total_timesteps=steps,
                    callback=[progress_callback, eval_callback, checkpoint_callback],
                    tb_log_name=run_name)

    print("Done! Saving model...")
    model.save("models/{}".format(run_name))

if __name__ == "__main__":
    run("dqn")