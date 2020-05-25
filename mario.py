import gym
from baselines import deepq

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# print("ACTION SPACE:", env.action_space)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    move = env.action_space.sample()
    state, reward, done, info = env.step(move)
    env.render()

env.close()

# 0 = still
# 1, 3 = right
# 2, 4 = jump, move right
# 5 = jump, stay still
# 6 = move left (backwards)
