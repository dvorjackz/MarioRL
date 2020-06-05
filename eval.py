import retro
import tensorflow as tf
import argparse
from nes_py.wrappers import JoypadSpace
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv


def eval(model_name):
    # Suppress warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, n_frames=4)

    env_render = gym_super_mario_bros.make('SuperMarioBros-v0')
    env_render = JoypadSpace(env_render, RIGHT_ONLY)

    model = DQN.load("models/{}".format(model_name))

    obs = env.reset()
    env_render.reset()

    cr = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env_render.step(action)
        cr += rewards
        print("reward:{}\t\t".format(cr))
        env_render.render()
        if (done):
            print("finished an episode")
    print(cr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluate which model')
    parser.add_argument('model', type=str,
                        help='required model name inside models folder')
    args = parser.parse_args()
    print("Evaluating model models/{}.zip".format(args.model))
    eval(args.model)
