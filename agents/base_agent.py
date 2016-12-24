import tensorflow as tf
import numpy as np
import cv2
import random
import threading
from replayMemory import ReplayMemory
import commonOps


class BaseAgent:

    def __init__(self, config, session):
        # build the net
        self.config = config
        self.sess = session
        self.RM = ReplayMemory(config)
        self.step_count = 0
        self.episode = 0
        self.isTesting = False
        self.game_state = np.zeros(
            (1, 84, 84, self.config.buff_size), dtype=np.uint8)
        self.reset_game()
        self.timeout_option = tf.RunOptions(timeout_in_ms=5000)

    def step(self, x, r):
        r = max(-1, min(1, r))
        if not self.isTesting:
            if not self.episode_begining:
                self.RM.add(
                    self.game_state[
                        :, :, :, -1], self.game_action, self.game_reward, False)
            else:
                self.episode_begining = False
            self.observe(x, r)
            self.game_action = self.e_greedy_action(self.epsilon())
            self.update()
            self.step_count += 1
        else:
            self.observe(x, r)
            self.game_action = self.e_greedy_action(0.01)
        return self.game_action

    # Add the transition to RM and reset the internal state for the next
    # episode
    def done(self):
        if not self.isTesting:
            self.RM.add(
                self.game_state[:, :, :, -1],
                self.game_action, self.game_reward, True)
        self.reset_game()

    def observe(self, x, r):
        self.game_reward = r
        x_ = cv2.resize(x, (84, 84))
        x_ = cv2.cvtColor(x_, cv2.COLOR_RGB2GRAY)
        self.game_state = np.roll(self.game_state, -1, axis=3)
        self.game_state[0, :, :, -1] = x_

    def e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        else:
            action = np.argmax(
                self.sess.run(
                    self.Q, feed_dict={
                        self.state_ph: self.game_state})[0])
        return action

    def testing(self, t=True):
        self.isTesting = t

    def set_action_mode(self, mode):
        if mode not in self.action_modes:
            raise Exception(mode+" is not a valid action mode")
        self.select_action = self.action_modes[mode]

    def reset_game(self):
        self.episode_begining = True
        self.game_state.fill(0)

    def epsilon(self):
        if self.step_count < self.config.exploration_steps:
            return self.config.initial_epsilon - \
                ((self.config.initial_epsilon - self.config.final_epsilon) /
                 self.config.exploration_steps) * self.step_count
        else:
            return self.config.final_epsilon

    def set_summary_writer(self, summary_writter):
        self.summary_writter = summary_writter
