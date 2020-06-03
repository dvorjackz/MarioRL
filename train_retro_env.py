from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.callbacks import CallbackList, EvalCallback
from callbacks import ProgressBarManager
import retro
from make_env import make_env
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# env = retro.RetroEnv(game='SuperMarioBros-Nes',
#                      use_restricted_actions=retro.Actions.DISCRETE)
num_cpu = 4  # Number of processes to use
# Create the vectorized environment
env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

# obs = env.reset()
# while True:
# 	obs, rew, done, info = env.step(env.action_space.sample())
# 	env.render()
# 	if done:
# 		obs = env.reset()
# env.close()

print ("Setting up environment...")
# # env = gym_super_mario_bros.make('SuperMarioBros-v3')
# # env = JoypadSpace(env, SIMPLE_MOVEMENT)

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
            learning_starts=1000,
            learning_rate=1e-4,
            exploration_final_eps=0.01,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
			train_freq=4,
            tensorboard_log="./mario_tensorboard/"
        )

print("Training starting...")
steps = 2000
with ProgressBarManager(steps) as progress_callback:
    model.learn(total_timesteps=steps,
                callback=[progress_callback],
                tb_log_name="run")

print("Done! Saving model...")
model.save("dqn")
