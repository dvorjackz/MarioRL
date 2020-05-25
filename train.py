from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from baselines import deepq

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=10,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10
    )
    print("Saving model to mario_model.pkl")
    act.save("mario_model.pkl")


if __name__ == '__main__':
    main()