# List of potential hyperparameters to tune

TIME_STEPS = 12000

# Data preprocessing & other Atari wrappers
FRAME_STACK = 4 # Nature uses 4
FRAME_SKIP = 8 # Nature uses 4

# Neural network
BATCH_SIZE = 96 # Nature uses 32, higher batch sizes according to https://arxiv.org/pdf/1803.02811.pdf)
LEARNING_RATE = 1e-4
EXPLORATION_FRACT = 0.1 # Nature uses 0.1
P_REPLAY_ALPHA = 0.6
TRAINING_FREQ = 4 # Nature uses 4: https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
TARGET_UPDATE_FREQ = 10000 # Nature uses 10000
