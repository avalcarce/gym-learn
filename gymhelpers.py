import math
import shutil
from scipy.misc import imresize, imsave


from .utils import *
from .agents import AgentEpsGreedy
from .valuefunctions import ValueFunctionDQN
from .datastructures import DoubleEndedQueue, SumTree
from plot_utils import shadow_plot

import sys
import psutil
import gym
from gym import wrappers
import copy
import time
from sys import platform
from textwrap import wrap
import matplotlib.pyplot as plt
if platform == "linux" or platform == "linux2":
    plt.switch_backend('Agg')  # This is to generate images without having a window appear.


class ExperimentsManager:
    def __init__(self, env_name, agent_value_function_hidden_layers_size, results_dir_prefix=None, summaries_path=None,
                 figures_dir=None, discount=0.99, decay_eps=0.995, eps_min=0.0001, epsilon_reheat=False,
                 conditional_epsilon_reheat=False,
                 learning_rate=1E-4, decay_lr=False,
                 learning_rate_end=None, max_step=10000, replay_memory_max_size=100000, ep_verbose=False,
                 exp_verbose=True, batch_size=64, upload_last_exp=False, double_dqn=False,
                 target_params_update_period_steps=1, gym_api_key="", gym_algorithm_id=None, checkpoints_dir='ChkPts',
                 min_avg_rwd=-110, replay_period_steps=1, per_proportional_prioritization=False,
                 per_apply_importance_sampling=False, per_alpha=0.6, per_beta0=0.4, render_environment=False,
                 checkpoint_save_period_steps=None, restoration_checkpoint=None, kpis_dir=None,
                 use_long_dirnames=False):
        self.env_name = env_name
        self.results_dir_prefix = results_dir_prefix
        self.render_environment = render_environment
        self.gym_stats_dir = None
        self.summaries_path = summaries_path
        self.summaries_path_current = summaries_path
        self.figures_dir = figures_dir
        self.discount = discount
        self.decay_eps = decay_eps
        self.eps_min = eps_min
        self.epsilon_reheat = epsilon_reheat
        self.conditional_epsilon_reheat = conditional_epsilon_reheat
        self.learning_rate = learning_rate
        self.decay_lr = decay_lr
        self.learning_rate_end = learning_rate_end
        self.max_step = max_step
        self.prob_sample_state = min([1, 2000 / (self.max_step * 1000)])  # Sample 2000 summaries of the current state
        self.n_ep = None
        self.replay_period_steps = replay_period_steps
        self.replay_memory_max_size = replay_memory_max_size
        self.ep_verbose = ep_verbose  # Whether or not to print progress during episodes
        self.exp_verbose = exp_verbose  # Whether or not to print progress during experiments
        self.upload_last_exp = upload_last_exp
        assert target_params_update_period_steps > 0, "The period for updating the target parameters must be positive."
        self.target_params_update_period_steps = target_params_update_period_steps
        self.gym_api_key = gym_api_key
        self.gym_algorithm_id = gym_algorithm_id
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir_current = checkpoints_dir
        self.checkpoint_save_period_steps = checkpoint_save_period_steps  # type: int
        self.restoration_checkpoint = restoration_checkpoint
        self.use_long_dirnames = use_long_dirnames
        self.kpis_dir = kpis_dir
        self.embeddings_metadata_file = None
        self.sprite_path = None
        self.performance_evaluation_mode = False

        # Prioritized Experience Replay parameters. See https://arxiv.org/pdf/1511.05952.pdf
        self.per_proportional_prioritization = per_proportional_prioritization  # Flavour of Prioritized Experience Rep.
        self.per_apply_importance_sampling = per_apply_importance_sampling
        self.per_alpha = per_alpha
        self.per_beta0 = per_beta0
        self.per_beta = self.per_beta0

        self.agent = None
        self.batch_size = batch_size
        self.agent_value_function_hidden_layers_size = agent_value_function_hidden_layers_size
        self.double_dqn = double_dqn

        self.global_step = 0  # Current step over all episodes
        self.step = 0  # Current step per episode
        self.ep = 0
        self.exp = 0

        self.min_avg_rwd = min_avg_rwd     # Minimum average reward to consider the problem as solved
        self.n_avg_ep = 100         # Number of consecutive episodes to calculate the average reward

        self.conf_msg = "\nEXECUTING EXPERIMENT {} OF {} IN ENVIRONMENT {}."
        self.episode_progress_msg = "Step {:5d}/{:5d}. Loss = {:3.2e}."
        self.exp_progress_msg = "Exp {:2d}. Ep {:5d}, Rwd={:1.4f} ({:1.4f}/{:1.4f}/\033[1m{:1.4f}\033[0m/{:1.4f} " + \
                                "over {:3d} episodes). {} exceeded in {:4d} eps. Loss={:1.2e} (avg={:1.2e}). " + \
                                "Agent epsilon={:3.2f} %. Avg episode: {:2.6f} ms."
        self.exps_conf_str = ""

        self.dpi = 400  # Plotting option

        # Memory pre-allocation
        self.Rwd_per_ep_v = np.zeros((1, 5000))
        self.Loss_per_ep_v = np.zeros((1, 5000))
        self.Avg_Rwd_per_ep = np.zeros((1, 5000))
        self.Avg_Loss_per_ep = np.zeros((1, 5000))
        self.n_eps_to_reach_min_avg_rwd = np.zeros(1, dtype=float)
        self.Agent_Epsilon_per_ep = np.zeros((1, 5000))
        self.agent_value_function = np.zeros((1, 1, self.max_step))
        self.rwd_exps_avg = np.mean(self.Rwd_per_ep_v, axis=0)  # Rwd averaged over all experiments
        self.rwd_exps_avg_ma = np.zeros(self.rwd_exps_avg.shape[0])
        self.rwd_exps_avg_movstd = np.zeros(self.rwd_exps_avg.shape[0])
        self.rwd_exps_avg_percentile5 = np.zeros(self.rwd_exps_avg.shape[0])
        self.rwd_exps_avg_percentile95 = np.zeros(self.rwd_exps_avg.shape[0])
        self.episode_duration_s = np.zeros(shape=(1, 1000), dtype=float)

    def __print_episode_progress(self, loss_v):
        if self.ep_verbose:
            if self.step > 0 and (self.step+1) % 20 == 0:
                print(self.episode_progress_msg.format(self.step, self.max_step, loss_v))

    def __print_experiment_progress(self):
        if self.exp_verbose:
            if self.ep % 8 == 0:
                rwd = self.Rwd_per_ep_v[self.exp, self.ep]
                ep0 = max(0, self.ep - 99)
                rwd_min = np.amin(self.Rwd_per_ep_v[self.exp, ep0:self.ep + 1])
                rwd_max = np.amax(self.Rwd_per_ep_v[self.exp, ep0:self.ep + 1])
                rwd_std = np.std(self.Rwd_per_ep_v[self.exp, ep0:self.ep + 1])
                avg_rwd = self.Avg_Rwd_per_ep[self.exp, self.ep]
                loss = self.Loss_per_ep_v[self.exp, self.ep]
                avg_loss = self.Avg_Loss_per_ep[self.exp, self.ep]

                avg_rwds = self.Avg_Rwd_per_ep[self.exp, 0:self.ep+1]
                idx_unsolved_eps = np.where(avg_rwds < self.min_avg_rwd)
                i_last_low_rwd = self.ep
                if idx_unsolved_eps[0].size > 0:
                    i_last_low_rwd = np.amax(idx_unsolved_eps)
                n_solved_eps = self.ep - i_last_low_rwd

                avg_episode_duration_ms = 0
                if self.ep > 0:
                    avg_episode_duration_ms = np.mean(self.episode_duration_s[self.exp, ep0:self.ep + 1]) * 1000

                print(
                    self.exp_progress_msg.format(self.exp, self.ep, rwd, rwd_min, rwd_max, avg_rwd, rwd_std,
                                                 self.n_avg_ep, self.min_avg_rwd, n_solved_eps, loss, avg_loss,
                                                 self.agent.eps*100, avg_episode_duration_ms))

    def __maybe_update_target_estimator(self):
        if self.global_step % self.target_params_update_period_steps == 0:
            self.agent.value_func.update_old_params()
            if self.ep_verbose:
                print("Copied model parameters to target network.")

    def run_episode(self, env, train=True):
        t = time.time()
        self.agent.state = env.reset()
        done = False
        self.agent.total_reward = 0
        loss_v = np.nan

        for self.step in range(self.max_step):
            if self.render_environment and self.step % 10 == 0:
                env.render()  # Render environment for human consumption
            self.__maybe_update_target_estimator()
            self.agent.anneal_per_importance_sampling(self.step, self.max_step)

            self.__print_episode_progress(loss_v)

            if done:
                break

            summaries_to_save = []
            if (self.global_step + 1) % 200*self.max_step == 0:
                summaries_to_save.append("progress")
            if np.random.uniform() <= self.prob_sample_state:
                summaries_to_save.append("state")

            save_embedding = self.global_step in self.embeddings_global_steps
            self.__maybe_collect_embedding_thumbnail(env, save_embedding)
            action = self.agent.act(self.global_step, state=self.agent.state, saveembedding=save_embedding,
                                    summaries_to_save=summaries_to_save)
            self.agent_value_function[self.exp, self.ep, self.step] = self.agent.current_value
            state_next, reward, done, info = env.step(action)
            self.agent.total_reward += reward

            self.agent.save_experience(self.agent.state, action, reward, state_next, done)
            if train and self.global_step % self.replay_period_steps == 0:
                summaries_to_save = []
                if self.global_step % 10000 == 0:
                    summaries_to_save.append("train")
                loss_v = self.agent.train_on_experience(self.batch_size, self.discount,
                                                        double_dqn=self.double_dqn, summaries_to_save=summaries_to_save)

            self.agent.state = copy.copy(state_next)

            if save_embedding:
                self.embeddings_metadata_file.write("{:2.6f}\n".format(self.agent.current_value))
            self.global_step += 1
        self.agent.value_func.update_summarizables(self.agent.total_reward, self.agent.eps, self.agent.current_value)
        self.episode_duration_s[self.exp, self.ep] = time.time() - t  # Time elapsed during this episode
        return loss_v, self.agent.total_reward

    def __maybe_collect_embedding_thumbnail(self, env, saveembedding=False):
        if saveembedding:
            env_img = env.render(mode="rgb_array")
            env_img = imresize(env_img, (self.embedding_thumbnail_h, self.embedding_thumbnail_w))
            (r, c) = self.nxt_thumb
            self.sprite_image[r:r + self.embedding_thumbnail_h, c:c + self.embedding_thumbnail_w] = env_img
            if self.nxt_thumb[1] == self.sprite_image.shape[1] - self.embedding_thumbnail_w:
                self.nxt_thumb[0] += self.embedding_thumbnail_h
            self.nxt_thumb[1] = (self.nxt_thumb[1] + self.embedding_thumbnail_w) % self.sprite_image.shape[1]
            self.n_saved_embeddings += 1

    def __maybe_stop_training(self, stop_training_min_avg_rwd, n_min_training_episodes, train):
        if stop_training_min_avg_rwd is not None:
            if train and self.ep >= n_min_training_episodes and \
                            self.Avg_Rwd_per_ep[self.exp, self.ep] >= stop_training_min_avg_rwd:
                train = False
                self.agent.explore = False
                print("Minimum average reward reached. Stop training and exploration.")
        return train

    def run_experiment(self, env, n_ep, stop_training_min_avg_rwd=None, n_min_training_episodes=100,
                       print_mean_episode_reward=False):
        self.n_ep = n_ep
        self.global_step = 0
        self.prob_sample_state = min([1, 2000 / (self.max_step * n_ep)])  # Sample 2000 summaries of the current state

        train = True
        # One experiment is composed of n_ep sequential episodes
        for self.ep in range(n_ep):
            loss_v, total_reward = self.run_episode(env, train)

            # Collect episode results
            self.Rwd_per_ep_v[self.exp, self.ep] = total_reward
            self.Loss_per_ep_v[self.exp, self.ep] = loss_v

            # Calculate episode statistics
            last_rwds = self.Rwd_per_ep_v[self.exp, np.maximum(self.ep - (self.n_avg_ep - 1), 0):self.ep+1]
            last_losses = self.Loss_per_ep_v[self.exp, np.maximum(self.ep - (self.n_avg_ep - 1), 0):self.ep+1]
            self.Avg_Rwd_per_ep[self.exp, self.ep] = np.mean(last_rwds)
            self.Avg_Loss_per_ep[self.exp, self.ep] = np.nanmean(last_losses)
            self.Agent_Epsilon_per_ep[self.exp, self.ep] = self.agent.eps

            if self.Avg_Rwd_per_ep[self.exp, self.ep] >= self.min_avg_rwd:
                self.n_eps_to_reach_min_avg_rwd[self.exp] = np.minimum(self.ep,
                                                                       self.n_eps_to_reach_min_avg_rwd[self.exp])
            train = self.__maybe_evaluate_performance_and_reheat(stop_training_min_avg_rwd, n_min_training_episodes,
                                                                 train)

            if self.agent.eps > self.eps_min:
                self.agent.eps *= self.decay_eps

            if self.ep > 0 and self.ep % (n_ep//4) == 0:  # Save KPIs progress four times during each experiment
                self.save_kpis("_{}".format(self.ep))

            self.__print_experiment_progress()
        self.agent.value_func.save()  # Save one final checkpoint at the end of training
        self.__maybe_save_sprite_image("_{}".format(self.agent.value_func.scope))
        if self.embeddings_metadata_file is not None:
            self.agent.value_func.save_embeddings(self.summaries_path_current+"_{}".format(self.agent.value_func.scope),
                                                  os.path.basename(self.embeddings_metadata_file.name),
                                                  self.sprite_path, self.embedding_thumbnail_w,
                                                  self.embedding_thumbnail_h)
        if print_mean_episode_reward:
            mean_reward = np.mean(self.Rwd_per_ep_v[self.exp, :])
            print("Average reward over all episodes: {}.".format(mean_reward))

    def __maybe_evaluate_performance_and_reheat(self, stop_training_min_avg_rwd, n_min_training_episodes, train):
        if not self.performance_evaluation_mode:
            if self.epsilon_reheat:
                if self.agent.eps <= self.eps_min:
                    if self.conditional_epsilon_reheat:
                        if self.Avg_Rwd_per_ep[self.exp, self.ep] < self.min_avg_rwd:
                            ep0 = np.maximum(self.ep - (self.n_avg_ep - 1), 0)
                            last_avg_losses = self.Avg_Loss_per_ep[self.exp, ep0:self.ep + 1]
                            if np.all(last_avg_losses < pow(self.min_avg_rwd * 0.2, 2)):  # If loss < 20% of target Rwd
                                train = False
                                self.__enable_performance_evaluation_mode()
                                print("Loss is low enough. Stopping training and exploration to evaluate performance.")
                    else:  # Unconditional epsilon reheat
                        self.__reheat_exploration()
            else:
                train = self.__maybe_stop_training(stop_training_min_avg_rwd, n_min_training_episodes, train)
        else:  # If in performance evaluation mode
            train = self.__evaluate_performance(train)
        return train

    def __enable_performance_evaluation_mode(self):
        self.agent.explore = False
        self.performance_evaluation_mode = True
        self.performance_evaluation_ep0 = self.ep

    def __evaluate_performance(self, train):
        last_perf_rwds = self.Rwd_per_ep_v[self.exp, self.performance_evaluation_ep0:self.ep + 1]
        self.Avg_Rwd_per_ep[self.exp, self.ep] = np.mean(last_perf_rwds)  # Overwrite Avg Rwd with perf Rwd
        n_perf_eps = self.ep - self.performance_evaluation_ep0
        if n_perf_eps > 500:
            last_perf_avg_rwds = self.Avg_Rwd_per_ep[self.exp, self.ep - (n_perf_eps // 4):self.ep + 1]
            reward_max = np.max(last_perf_avg_rwds)
            reward_min = np.min(last_perf_avg_rwds)
            if reward_max - reward_min < self.Avg_Rwd_per_ep[self.exp, self.ep] / 20:  # Perf. eval. complete
                print("The average reward is {}.".format(self.Avg_Rwd_per_ep[self.exp, self.ep]))
                if self.epsilon_reheat and self.Avg_Rwd_per_ep[self.exp, self.ep] < self.min_avg_rwd:
                    self.__reheat_exploration()
                print("Resuming training and exploration.")
                train = True
                self.agent.explore = True
                self.performance_evaluation_mode = False
        return train

    def __reheat_exploration(self):
        self.agent.eps = 1.0
        print("Reheating exploration.")

    def __create_gym_stats_directory(self, env):
        if self.results_dir_prefix is None:
            raise ValueError("A prefix for the Gym results directory must be provided.")
        if not os.path.exists(self.results_dir_prefix):
            os.makedirs(self.results_dir_prefix)
        t = get_last_folder_id(self.results_dir_prefix) + 1  # Calculate next test id
        self.gym_stats_dir = os.path.join(self.results_dir_prefix, str(t).zfill(4))
        if not os.path.exists(self.gym_stats_dir):
            os.makedirs(self.gym_stats_dir)
        else:
            raise FileExistsError(self.gym_stats_dir)
        return wrappers.Monitor(env, self.gym_stats_dir)

    def __build_layers_size_str(self, state_dim, n_actions):
        layers_size = str(state_dim)
        for s in self.agent_value_function_hidden_layers_size:
            layers_size += "-" + str(s)
        layers_size += "-" + str(n_actions)

        return layers_size

    def __build_experiments_conf_str(self, n_ep, n_actions, state_dim):

        layers_size = self.__build_layers_size_str(state_dim, n_actions)

        if self.use_long_dirnames:
            exp_conf_str = "{}_{}_{:1.2f}_{:1.4f}_{:1.2e}_{:1.0e}_DL{}_" +\
                           "DDQN{}_{}_p{}_C{}_K{}_PER{}_IS{}_a{:1.1f}_b0{:1.1f}"
            self.exps_conf_str = exp_conf_str.format(time.strftime("%Y%m%d%H%M%S"), layers_size, self.discount,
                                                     self.decay_eps, self.eps_min, self.learning_rate,
                                                     1 if self.decay_lr else 0, 1 if self.double_dqn else 0,
                                                     self.batch_size, n_ep,
                                                     self.target_params_update_period_steps, self.replay_period_steps,
                                                     1 if self.per_proportional_prioritization else 0,
                                                     1 if self.per_apply_importance_sampling else 0,
                                                     self.per_alpha, self.per_beta0)
        else:
            self.exps_conf_str = "{}".format(time.strftime("%Y%m%d%H%M%S"))

    def __save_config_file(self, n_exps, n_ep, state_dim, n_actions, stop_training_min_avg_rwd):
        config_file_dir = os.path.join("Logs", self.exps_conf_str)
        if not os.path.exists(config_file_dir):
            os.makedirs(config_file_dir)

        config_file = open(os.path.join(config_file_dir, "config.txt"), "w")

        rest_chkpt_folder = "N.K."
        rest_chkpt_file = "N.K."
        if self.restoration_checkpoint is not None:
            head, rest_chkpt_file = os.path.split(self.restoration_checkpoint)
            rest_chkpt_folder = os.path.basename(head)

        layers_size = self.__build_layers_size_str(state_dim, n_actions)
        exp_conf_str = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
        config_file.write(exp_conf_str.format(self.env_name, self.exps_conf_str[:14], layers_size,
                                              self.target_params_update_period_steps,
                                              self.discount, self.decay_eps, self.eps_min, self.learning_rate,
                                              "TRUE" if self.decay_lr else "FALSE",
                                              str(self.learning_rate_end) if self.decay_lr else "N.A.",
                                              "TRUE" if self.epsilon_reheat else "FALSE",
                                              self.max_step,
                                              "TRUE" if self.double_dqn else "FALSE", self.replay_memory_max_size,
                                              self.batch_size, n_exps, n_ep, self.replay_period_steps,
                                              stop_training_min_avg_rwd, self.min_avg_rwd,
                                              self.per_proportional_prioritization, self.per_apply_importance_sampling,
                                              self.per_alpha, self.per_beta0,
                                              rest_chkpt_folder, rest_chkpt_file))
        config_file.close()

    def __create_figures_directory(self):
        if self.figures_dir is not None:
            self.figures_dir = os.path.join(self.figures_dir, self.env_name, self.exps_conf_str)
            if not os.path.exists(self.figures_dir):
                os.makedirs(self.figures_dir)
            else:
                for dirpath, dirnames, files in os.walk(self.figures_dir):
                    if files:
                        raise FileExistsError("The figures directory exists and has files: {}".format(self.figures_dir))
                    else:
                        break

    def __create_kpis_directory(self):
        if self.kpis_dir is not None:
            self.kpis_dir = os.path.join(self.kpis_dir, self.env_name, self.exps_conf_str)
            if not os.path.exists(self.kpis_dir):
                os.makedirs(self.kpis_dir)
            else:
                for dirpath, dirnames, files in os.walk(self.kpis_dir):
                    if files:
                        raise FileExistsError("The kpis directory exists and has files: {}".format(self.kpis_dir))
                    else:
                        break

    def __create_embeddings_metadata_file(self, collect_embeddings, suffix=""):
        if collect_embeddings:
            self.embeddings_metadata_file = open(os.path.join(self.summaries_path_current+suffix, "metadata.tsv"), "w")
            self.nxt_thumb = np.array([0, 0])  # Next pixels in the sprite image onto which to store the next thumbnail.
            self.n_saved_embeddings = 0

    def __close_embeddings_metadata_file(self):
        if self.embeddings_metadata_file is not None:
            self.embeddings_metadata_file.close()

    def __maybe_save_sprite_image(self, suffix=""):
        if self.embeddings_metadata_file is not None:
            self.sprite_path = os.path.join(self.summaries_path_current+suffix, "embeddings_sprite.jpg")
            imsave(self.sprite_path, self.sprite_image)

    def get_environment_actions(self, env):
        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError("Continuous action spaces are not supported yet.")
        elif isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n
        else:
            raise NotImplementedError("{} action spaces are not supported yet.".format(type(env.action_space)))
        return n_actions

    def __update_summaries_path_current(self):
        if self.summaries_path is not None:
            self.summaries_path_current = os.path.join(self.summaries_path, self.env_name,
                                                       self.exps_conf_str + "_Exp" + str(self.exp))

    def __update_checkpoints_dir_current(self):
        if self.checkpoints_dir is not None:
            self.checkpoints_dir_current = os.path.join(self.checkpoints_dir, self.env_name,
                                                        self.exps_conf_str + "_Exp" + str(self.exp))
            if not os.path.exists(self.checkpoints_dir_current):
                os.makedirs(self.checkpoints_dir_current)

    def __get_problem_dimensionality(self):
        env = gym.make(self.env_name)
        n_actions = self.get_environment_actions(env)
        state_dim = env.observation_space.high.shape[0]
        self.memory_check(env)
        env.close()
        return n_actions, state_dim

    def __update_embeddings_configuration(self, n_ep, n_embeddings=0):
        self.embeddings_global_steps = np.random.choice(n_ep * self.max_step, size=n_embeddings, replace=False)
        assert n_embeddings <= 10000, "Cannot collect more than 10000 embeddings."
        self.embeddings_sprite_n_rows = math.ceil(math.sqrt(n_embeddings)) if n_embeddings > 0 else 0
        self.embedding_thumbnail_h = 8192 // self.embeddings_sprite_n_rows if n_embeddings > 0 else 0
        self.embedding_thumbnail_w = 8192 // self.embeddings_sprite_n_rows if n_embeddings > 0 else 0
        self.sprite_image = np.zeros((self.embeddings_sprite_n_rows * self.embedding_thumbnail_h,
                                      self.embeddings_sprite_n_rows * self.embedding_thumbnail_w, 3), dtype=np.uint8)

    def run_experiments(self, n_exps, n_ep, stop_training_min_avg_rwd=None, plot_results=True, figures_format=None,
                        agent=None, n_embeddings=0, n_min_training_episodes=100, print_mean_episode_reward=False):
        self.Rwd_per_ep_v = np.zeros((n_exps, n_ep))
        self.Loss_per_ep_v = np.zeros((n_exps, n_ep))
        self.Avg_Rwd_per_ep = np.zeros((n_exps, n_ep))
        self.n_eps_to_reach_min_avg_rwd = np.zeros(n_exps, dtype=float)
        self.n_eps_to_reach_min_avg_rwd.fill(n_ep)
        self.Avg_Loss_per_ep = np.zeros((n_exps, n_ep))
        self.Agent_Epsilon_per_ep = np.zeros((n_exps, n_ep))
        self.agent_value_function = np.zeros((n_exps, n_ep, self.max_step))
        self.episode_duration_s = np.zeros(shape=(n_exps, n_ep), dtype=float)
        self.__update_embeddings_configuration(n_ep, n_embeddings)

        n_actions, state_dim = self.__get_problem_dimensionality()

        self.__build_experiments_conf_str(n_ep, n_actions, state_dim)
        self.__create_figures_directory()
        self.__create_kpis_directory()
        self.__save_config_file(n_exps, n_ep, state_dim, n_actions, stop_training_min_avg_rwd)

        if self.checkpoint_save_period_steps is None:
            self.checkpoint_save_period_steps = n_ep*self.max_step//4  # By default, save 4 checkpoints
        ckpt_sv_period_epochs = self.checkpoint_save_period_steps // self.replay_period_steps

        for self.exp in range(n_exps):
            print(self.conf_msg.format(self.exp, n_exps, self.env_name))
            if self.use_long_dirnames:
                print(self.exps_conf_str)

            env = gym.make(self.env_name)  # Create new environment
            env.seed(self.exp)
            assert state_dim == env.observation_space.high.shape[0]

            if self.upload_last_exp and self.exp == n_exps-1:
                env = self.__create_gym_stats_directory(env)

            self.__update_summaries_path_current()
            self.__update_checkpoints_dir_current()

            # Create agent
            if agent is None:
                value_function = ValueFunctionDQN(scope="q", state_dim=state_dim, n_actions=n_actions,
                                                  train_batch_size=self.batch_size, learning_rate=self.learning_rate,
                                                  hidden_layers_size=self.agent_value_function_hidden_layers_size,
                                                  decay_lr=self.decay_lr, learning_rate_end=self.learning_rate_end,
                                                  n_lr_decay_epochs=n_ep*self.max_step/self.replay_period_steps,
                                                  huber_loss=False, summaries_path=self.summaries_path_current,
                                                  reset_default_graph=True,
                                                  checkpoints_dir=self.checkpoints_dir_current,
                                                  checkpoint_save_period_epochs=ckpt_sv_period_epochs,
                                                  apply_wis=self.per_apply_importance_sampling,
                                                  restoration_checkpoint=self.restoration_checkpoint,
                                                  n_embeddings=n_embeddings, epsilon0=1.0)
                self.__create_embeddings_metadata_file(n_embeddings > 0, "_{}".format(value_function.scope))

                self.agent = AgentEpsGreedy(n_actions=n_actions, value_function_model=value_function, eps=1.0,
                                            per_proportional_prioritization=self.per_proportional_prioritization,
                                            per_apply_importance_sampling=self.per_apply_importance_sampling,
                                            per_alpha=self.per_alpha, per_beta0=self.per_beta0)

                if self.per_proportional_prioritization:
                    self.agent.memory = SumTree(self.replay_memory_max_size)
                else:
                    self.agent.memory = DoubleEndedQueue(max_size=self.replay_memory_max_size)
            else:
                self.agent = agent

            self.run_experiment(env, n_ep, stop_training_min_avg_rwd, n_min_training_episodes,
                                print_mean_episode_reward)   # Action happens here

            if agent is None:
                value_function.close_summary_file()
                self.__close_embeddings_metadata_file()

            env.close()
            if self.upload_last_exp and self.exp == n_exps - 1:
                print("Trying to upload results to the scoreboard.")
                gym.upload(self.gym_stats_dir, api_key=self.gym_api_key, algorithm_id=self.gym_algorithm_id)

            # Plot results
            self.plot_rwd_loss(figures_format=figures_format)
            self.plot_value_function(figures_format=figures_format)
            self.print_experiment_summary()

        self.save_kpis()
        self.calculate_avg_rwd()
        self.plot_rwd_averages(n_exps, figures_format=figures_format)
        self.print_summary()
        if plot_results:
            plt.show()

        # Return the final Rwd averaged over all experiments AND the mean number of episodes needed to reach the min Rwd
        return self.rwd_exps_avg_ma[-1], np.mean(self.n_eps_to_reach_min_avg_rwd)

    def memory_check(self, env):
        state_next, reward, done, info = env.step(0)
        exp_size = sys.getsizeof(state_next) * 2 + sys.getsizeof(reward) + sys.getsizeof(done) + sys.getsizeof(int)
        needed_mem_bytes = exp_size * self.replay_memory_max_size
        mem = psutil.virtual_memory()
        if mem.available < needed_mem_bytes:
            max_mem_size_av = mem.available//exp_size
            max_mem_size_tot = mem.total // exp_size
            explanation = "The replay memory size exceeds the available memory." +\
                          " Each (s, a, r, s', done) tuple occupies {:2.1f} KB.\n" +\
                          " A replay memory of size {} would require " +\
                          "{:2.1f} GB, which exceeds the available (total) {:2.1f} ({:2.1f}) GB.\n" +\
                          " Given the available memory, a replay memory size of {} (max: {}) may do the job."
            raise ValueError(explanation.format(exp_size/1024, self.replay_memory_max_size,
                                                needed_mem_bytes/(1024 ** 3), mem.available/(1024 ** 3),
                                                mem.total/(1024 ** 3), max_mem_size_av, max_mem_size_tot))

    def print_experiment_summary(self):
        duration_ms = np.mean(self.episode_duration_s[self.exp, :]) * 1000
        print("Average episode duration: {:2.6f} ms".format(duration_ms))

    def calculate_avg_rwd(self):
        self.rwd_exps_avg = np.mean(self.Rwd_per_ep_v, axis=0)  # Rwd averaged over all experiments
        self.rwd_exps_avg_ma = np.zeros(self.rwd_exps_avg.shape[0])
        self.rwd_exps_avg_movstd = np.zeros(self.rwd_exps_avg.shape[0])
        self.rwd_exps_avg_percentile5 = np.zeros(self.rwd_exps_avg.shape[0])
        self.rwd_exps_avg_percentile95 = np.zeros(self.rwd_exps_avg.shape[0])

        for s in range(self.rwd_exps_avg.shape[0]):
            self.rwd_exps_avg_ma[s] = np.mean(self.rwd_exps_avg[max(0, s - 99):s + 1])
            self.rwd_exps_avg_movstd[s] = np.std(self.rwd_exps_avg[max(0, s - 99):s + 1])
            self.rwd_exps_avg_percentile5[s] = np.percentile(self.rwd_exps_avg[max(0, s - 99):s + 1], 5)
            self.rwd_exps_avg_percentile95[s] = np.percentile(self.rwd_exps_avg[max(0, s - 99):s + 1], 95)

    def plot_rwd_averages(self, n_exps, figures_format=None):
        n_ep = self.Rwd_per_ep_v.shape[1]
        eps = range(n_ep)

        if self.figures_dir is not None:
            # PLOT ALL EXPERIMENTS
            fig = plt.figure()
            for i in range(n_exps):
                shadow_plot(eps, self.Rwd_per_ep_v[i, :], label="Exp {}".format(i), smooth=0.1)
            plt.xlabel("Episode number")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.legend(loc='upper left')

            ttl = "Average reward. " + self.exps_conf_str
            plt.title("\n".join(wrap(ttl, 60)))

            if self.figures_dir is not None:
                fig_savepath = os.path.join(self.figures_dir, "RwdsExpsComp.png")
                plt.savefig(fig_savepath, dpi=self.dpi)

                if figures_format is not None:
                    try:
                        fig_savepath = os.path.join(self.figures_dir,
                                                    "RwdsComparisonsAcrossExps.{}".format(figures_format))
                        plt.savefig(fig_savepath, format=figures_format, dpi=self.dpi)
                    except:
                        print("Error while saving figure in {} format.".format(figures_format))
            plt.close(fig)

            # PLOT AVERAGE OVER ALL EXPERIMENTS
            fig = plt.figure()
            plt.subplot(211)
            plt.plot(eps, self.rwd_exps_avg, label="Average over {:3d} experiments".format(n_exps))
            plt.ylabel("Reward per episode")
            plt.grid(True)

            plt.plot(eps, self.rwd_exps_avg_percentile95, label="95th percentile over 100 episodes")
            plt.plot(eps, self.rwd_exps_avg_ma, label="100-episode moving average")
            plt.plot(eps, self.rwd_exps_avg_percentile5, label="5th percentile over 100 episodes")
            plt.legend(loc='lower right')
            plt.title("Final average reward: {:3.2f} (std={:3.2f})".format(self.rwd_exps_avg_ma[-1],
                                                                           self.rwd_exps_avg_movstd[-1]))

            loss_exps_avg = np.mean(self.Loss_per_ep_v, axis=0)
            plt.subplot(212)
            plt.semilogy(eps, loss_exps_avg, label="Average over {:3d} experiments".format(n_exps))
            plt.xlabel("Episode number")
            plt.ylabel("Loss per episode")
            plt.grid(True)

            loss_exps_avg_ma = np.zeros(loss_exps_avg.shape[0])
            for s in range(loss_exps_avg.shape[0]):
                loss_exps_avg_ma[s] = np.mean(loss_exps_avg[max(0, s - 100):s + 1])
            plt.plot(eps, loss_exps_avg_ma, label="100-episode moving average")
            plt.legend(loc='lower right')

            plt.suptitle("\n".join(wrap(self.exps_conf_str, 60)))
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)

            if self.figures_dir is not None:
                fig_savepath = os.path.join(self.figures_dir, "ExpsAverage.png")
                plt.savefig(fig_savepath, dpi=self.dpi)

                if figures_format is not None:
                    try:
                        fig_savepath = os.path.join(self.figures_dir, "ExpsAverage.{}".format(figures_format))
                        plt.savefig(fig_savepath, format=figures_format, dpi=self.dpi)
                    except:
                        print("Error while saving figure in {} format.".format(figures_format))
            plt.close(fig)

    def print_summary(self):
        n_eps = np.argmax(self.rwd_exps_avg_ma >= self.min_avg_rwd)
        print("Average final reward: {:3.2f} (std={:3.2f}).\n".format(self.rwd_exps_avg_ma[-1],
                                                                      self.rwd_exps_avg_movstd[-1]))
        if n_eps is None:
            print("The 100-episode moving average never reached {}.".format(self.min_avg_rwd))
        else:
            print("The 100-episode moving average reached {} after {} episodes.".format(self.min_avg_rwd, n_eps))

    def plot_value_function(self, figures_format=None):
        if self.figures_dir is not None:
            n_ep = self.Rwd_per_ep_v.shape[1]
            fig = plt.figure()
            for ep in draw_equispaced_items_from_sequence(7, n_ep):
                plt.plot(self.agent_value_function[self.exp, ep, :], label="Episode {:4d}".format(ep))
            plt.xlabel("Steps")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend(loc='lower right')
            plt.title("Value functions for experiment {:2d}".format(self.exp))

            if self.figures_dir is not None:
                fig_savepath = os.path.join(self.figures_dir, "Exp{}_ValueFuncs.png".format(self.exp))
                plt.savefig(fig_savepath, dpi=self.dpi)
                if figures_format is not None:
                    try:
                        fig_savepath = os.path.join(self.figures_dir,
                                                    "Exp{}_ValueFuncs.{}".format(self.exp, figures_format))
                        plt.savefig(fig_savepath, format=figures_format, dpi=self.dpi)
                    except:
                        print("Error while saving figure in {} format.".format(figures_format))
            plt.close(fig)

    def plot_rwd_loss(self, figures_format=None):
        if self.figures_dir is not None:
            n_ep = self.Rwd_per_ep_v.shape[1]

            eps = range(n_ep)
            fig = plt.figure()
            ax1 = plt.subplot(211)
            shadow_plot(eps, self.Rwd_per_ep_v[self.exp, :], smooth=0.1)
            plt.xlabel("Episode number")
            plt.ylabel("Reward per episode")

            ax2 = ax1.twinx()
            plt.plot(eps, self.Agent_Epsilon_per_ep[self.exp, :], label="Agent epsilon", color='r')
            ax2.set_ylabel(r'Agent $\varepsilon$', color='r')
            ax2.tick_params('y', colors='r')

            plt.grid(True)
            ttl = "Final average reward: {:3.2f} (SD={:3.2f})"
            plt.title(ttl.format(self.Avg_Rwd_per_ep[self.exp, -1],
                                 np.std(self.Rwd_per_ep_v[self.exp, n_ep-100:n_ep-1])))
            plt.legend(loc='lower right')

            rwd_per_ep_exp_avg = np.mean(self.Rwd_per_ep_v[0:self.exp+1, n_ep-100:n_ep-1], axis=1)
            print("Final mean reward, averaged over {} experiment{}: {} (std = {}).".format(self.exp+1,
                                                                                            's' if self.exp > 0 else '',
                                                                                            np.mean(rwd_per_ep_exp_avg),
                                                                                            np.std(rwd_per_ep_exp_avg)))
            plt.subplot(212)
            shadow_plot(eps, self.Loss_per_ep_v[self.exp, :], semilogy=True, smooth=0.1)
            plt.xlabel("Episode number")
            plt.ylabel("Loss per episode")
            plt.grid(True)
            plt.title("Value function loss")

            sttl = self.exps_conf_str + ". Experiment {}".format(self.exp)
            plt.suptitle("\n".join(wrap(sttl, 60)))
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)

            if self.figures_dir is not None:
                fig_savepath = os.path.join(self.figures_dir, "Experiment{}_Rwd_Loss.png".format(self.exp))
                plt.savefig(fig_savepath, dpi=self.dpi)

                if figures_format is not None:
                    try:
                        fig_savepath = os.path.join(self.figures_dir, "Experiment{}_Rwd_Loss.{}".format(self.exp,
                                                                                                        figures_format))
                        plt.savefig(fig_savepath, format=figures_format, dpi=self.dpi)
                    except:
                        print("Error while saving figure in {} format.".format(figures_format))
            plt.close(fig)

    def save_kpis(self, suffix=""):
        if self.kpis_dir is not None:
            shutil.rmtree(self.kpis_dir)  # To prevent lack of space issues, first delete all previous kpis.
            os.makedirs(self.kpis_dir)
            np.save(os.path.join(self.kpis_dir, "rwd_per_ep"+suffix), self.Rwd_per_ep_v)
            np.save(os.path.join(self.kpis_dir, "loss_per_ep"+suffix), self.Loss_per_ep_v)
            np.save(os.path.join(self.kpis_dir, "agent_value_function"+suffix), self.agent_value_function[:, ::100, :])
