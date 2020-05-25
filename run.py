from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from baselines import deepq

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="mario_model.pkl")

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(act(state[None])[0])
    env.render()

env.close()

  
# import gym

# from baselines import deepq


# def main():
#     env = gym.make("CartPole-v0")
#     act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="cartpole_model.pkl")

#     while True:
#         obs, done = env.reset(), False
#         episode_rew = 0
#         while not done:
#             env.render()
#             obs, rew, done, _ = env.step(act(obs[None])[0])
#             episode_rew += rew
#         print("Episode reward", episode_rew)


# if __name__ == '__main__':
#     main()