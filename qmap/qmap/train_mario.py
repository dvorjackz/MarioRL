import time
import tensorflow as tf
from baselines.common.schedules import LinearSchedule
from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds
import argparse
from .agents.models import ConvDeconvMap, ConvMlp
from .agents.q_map_dqn_agent import Q_Map_DQN_Agent
from .agents.replay_buffers import DoublePrioritizedReplayBuffer
from .envs.custom_mario import CustomSuperMarioAllStarsEnv
from .envs.wrappers import PerfLogger


# parser = argparse.ArgumentParser(
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument(
#     '--seed', help='random number generator seed', type=int, default=0)
# parser.add_argument('--path', default='qmap_results')
# parser.add_argument('--level', help='game level', default='1.1')
# parser.add_argument('--load', help='path to the agent to load', default=None)
# boolean_flag(parser, 'dqn', default=True)
# boolean_flag(parser, 'qmap', default=True)
# boolean_flag(parser, 'render', help='play the videos', default=False)
# args = parser.parse_args()


class QMapDQNMario(object):
    def __init__(self,
                 n_steps=int(5e4),
                 task_gamma=0.99,
                 seed=0,
                 path='qmap_results',
                 level='1.1',
                 load=None,
                 dqn=True,
                 qmap=True,
                 render=False):
        self.env = CustomSuperMarioAllStarsEnv(screen_ratio=4, coords_ratio=8, use_color=False,
                                               use_rc_frame=False, stack=3, frame_skip=2, action_repeat=4, level=level)
        self.coords_shape = self.env.coords_shape
        set_global_seeds(seed)
        self.env.seed(seed)

        print('~~~~~~~~~~~~~~~~~~~~~~')
        print(self.env)
        print(self.env.unwrapped.name)
        print('observations:', self.env.observation_space.shape)
        print('coords:     ', self.coords_shape)
        print('actions:    ', self.env.action_space.n)
        print('~~~~~~~~~~~~~~~~~~~~~~')

        def call():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess.__enter__()

            # Q-map
            if qmap:
                q_map_model = ConvDeconvMap(
                    convs=[(32, 8, 2), (32, 6, 2), (64, 4, 2)],
                    middle_hiddens=[1024],
                    deconvs=[(64, 4, 2), (32, 6, 2),
                             (self.env.action_space.n, 4, 1)],
                    coords_shape=self.coords_shape,
                    dueling=True,
                    layer_norm=True,
                    activation_fn=tf.nn.elu
                )
                q_map_random_schedule = LinearSchedule(
                    schedule_timesteps=n_steps, initial_p=0.1, final_p=0.05)
            else:
                q_map_model = None
                q_map_random_schedule = None

            # DQN
            if dqn:
                dqn_model = ConvMlp(
                    convs=[(32, 8, 2), (32, 6, 2), (64, 4, 2)],
                    hiddens=[1024],
                    dueling=True,
                )
                exploration_schedule = LinearSchedule(
                    schedule_timesteps=n_steps, initial_p=1.0, final_p=0.05)
            else:
                dqn_model = None
                exploration_schedule = LinearSchedule(
                    schedule_timesteps=n_steps, initial_p=1.0, final_p=1.0)

            if q_map_model is not None or dqn_model is not None:
                double_replay_buffer = DoublePrioritizedReplayBuffer(
                    int(5e5), alpha=0.6, epsilon=1e-6, timesteps=n_steps, initial_p=0.4, final_p=1.0)
            else:
                double_replay_buffer = None

            agent_name = '{}M'.format(n_steps//int(1e6))
            agent = Q_Map_DQN_Agent(
                observation_space=self.env.observation_space,
                n_actions=self.env.action_space.n,
                coords_shape=self.env.unwrapped.coords_shape,
                double_replay_buffer=double_replay_buffer,
                task_gamma=task_gamma,
                exploration_schedule=exploration_schedule,
                seed=seed,
                path=path,
                learning_starts=1000,
                train_freq=4,
                print_freq=1,
                env_name=self.env.unwrapped.name,
                agent_name=agent_name,
                renderer_viewer=render,
                # DQN
                dqn_q_func=dqn_model,
                dqn_lr=1e-4,
                dqn_optim_iters=1,
                dqn_batch_size=32,
                dqn_target_net_update_freq=1000,
                dqn_grad_norm_clip=1000,
                # Q-Map
                q_map_model=q_map_model,
                q_map_random_schedule=q_map_random_schedule,
                q_map_greedy_bias=0.5,
                q_map_timer_bonus=0.5,  # 50% more time than predicted
                q_map_lr=3e-4,
                q_map_gamma=0.9,
                q_map_n_steps=1,
                q_map_batch_size=32,
                q_map_optim_iters=1,
                q_map_target_net_update_freq=1000,
                q_map_min_goal_steps=15,
                q_map_max_goal_steps=30,
                q_map_grad_norm_clip=1000
            )
            if load is not None:
                agent.load(load)
            agent.seed(seed)

            self.env = PerfLogger(self.env, agent.task_gamma, agent.path)

            done = True
            episode = 0
            score = None
            best_score = -1e6
            best_distance = -1e6
            previous_time = time.time()
            last_ips_t = 0

            for t in range(n_steps+1):
                if done:
                    new_best = False
                    if episode > 0:
                        if score >= best_score:
                            best_score = score
                            new_best = True
                        distance = self.env.unwrapped.full_c
                        if distance >= best_distance:
                            best_distance = distance
                            new_best = True

                    if episode > 0 and (episode < 50 or episode % 10 == 0 or new_best):
                        current_time = time.time()
                        ips = (t - last_ips_t) / (current_time - previous_time)
                        print('step: {} IPS: {:.2f}'.format(t+1, ips))
                        name = 'score_%08.3f' % score + '_distance_' + \
                            str(distance) + '_steps_' + str(t+1) + \
                            '_episode_' + str(episode)
                        agent.renderer.render(name)
                        previous_time = current_time
                        last_ips_t = t
                    else:
                        agent.renderer.reset()

                    episode += 1
                    score = 0

                    ob = self.env.reset()
                    ac = agent.reset(ob)

                ob, rew, done, _ = self.env.step(ac)
                score += rew

                ac = agent.step(ob, rew, done)

        self.call = call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
