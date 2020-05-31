from nes_py.wrappers import JoypadSpace
from stable_baselines.common.vec_env import DummyVecEnv
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
            learning_starts=2000,
            learning_rate=1e-4,
            exploration_fraction=0.5,
            exploration_final_eps=0.01,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            train_freq=4,
            target_network_update_freq=1000,
            # tensorboard_log="./mario_tensorboard/"
        )

print("Training starting...")
steps = 5000
with ProgressBarManager(steps) as progress_callback:
    model.learn(total_timesteps=steps,
                callback=[progress_callback, eval_callback],
                tb_log_name="run")

print("Done! Saving model...")
model.save("dqn")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("reward:", rewards)
    print("standstill:{}\tx:{}".format(info['standstill'], info['x_pos']))
    env.render()
