import gym
from baselines import deepq
from baselines.common import models

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# model = deepq.models.cnn_to_mlp(
#     convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
#     hiddens=[256],
#     dueling=args.dueling
# )


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    act = deepq.learn(
        env,
        network=models.mlp(num_hidden=64, num_layers=1),
        lr=5e-4,
        max_timesteps=100,
        buffer_size=10000,
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=100,
        gamma=0.99,
        prioritized_replay=False,
        # callback=callback,
        print_freq=1
    )
    print("Saving model to smb_model.pkl")
    act.save("smb_model.pkl")


if __name__ == '__main__':
    main()
