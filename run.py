from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from baselines import deepq

from stable_baselines.deepq.policies import LnMlpPolicy, LnCnnPolicy
from stable_baselines import DQN

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
    env = JoypadSpace(env, RIGHT_ONLY)
    # act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="mario_model.pkl")

    model = DQN.load("mario_mdl")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        env.render()
  
if __name__ == '__main__':
    main()