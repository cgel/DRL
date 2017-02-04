import tensorflow as tf
import gym
import time
import string
import os
import re
from ale_python_interface import ALEInterface
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-replay_memory_capacity", type=int, default=1000000)
parser.add_argument("-steps_before_training", type=int, default=12500)
parser.add_argument("-exploration_steps", type=int, default=250000)
parser.add_argument("-sync_rate", type=int, default=2500)
parser.add_argument("-device", default="0")
parser.add_argument("-gamma", type=float, default=0.99)
parser.add_argument("-learning_rate", type=float, default=0.00025)
parser.add_argument("-initial_epsilon", type=float, default=1.)
parser.add_argument("-final_epsilon", type=float, default=0.1)
parser.add_argument("-testing_epsilon", type=float, default=0.05)
parser.add_argument("-buff_size", type=float, default=4)
parser.add_argument("-load_checkpoint", default="")
parser.add_argument("-agent", default="DQN")
parser.add_argument("-logging", default="")
parser.add_argument("-transition_function", default="oh_concat")
parser.add_argument("-env_name", default="Breakout-v0")
parser.add_argument("-alpha", type=float, default=0.9)
parser.add_argument("-update_summary_rate", type=int, default=50000)
config = parser.parse_args()
config.num_episodes = 100000
config.log_online_summary_rate = 100
config.log_perf_summary_rate = 1000
config.save_rate = 1000
config.log_percent_rate = 1000
config.test_run_num = 20
config.logging = config.logging not in ["0", "false", "False"]
config.device = "/gpu:"+config.device
print("Using agent "+config.agent)
import importlib
Agent = getattr(importlib.import_module("agents."+config.agent), config.agent)

print("Logging: " + str(config.logging))
if config.transition_function not in [
        "oh_concat", "expanded_concat", "conditional"]:
    raise Exception(config.transition_function+" is not valid transition function")
tf.logging.set_verbosity(tf.logging.ERROR)

if len(config.env.split("-")) == 1:
    # if env does not end in -v0 use alewrap
    class Env:
        def __init__(self):
            self.ale = ALEInterface()
            rom_name = "roms/Breakout.bin"
            self.ale.setInt("frame_skip", 4)
            self.ale.loadROM(rom_name)
            legal_actions = self.ale.getMinimalActionSet()
            self.action_map = {}
            for i in range(len(legal_actions)):
                self.action_map[i] = legal_actions[i]
            self.action_num = len(self.action_map)

        def reset(self):
            state = np.zeros((84, 84, 3), dtype=np.uint8)
            self.ale.reset_game()
            return state

        def step(self, action):
            reward = self.ale.act(self.action_map[action])
            state = self.ale.getScreenRGB()
            done = self.ale.game_over()
            return state, reward, done, ""
    env = Env()
    config.action_num = env.action_num
else:
    # else use gym
    env = gym.make(config.env_name)
    config.action_num = env.action_space.n

sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)

if config.logging:
    int_folders = []
    for folder in os.listdir("log"):
        folder_num = re.search(r"\d+", folder)
        if folder_num:
            int_folders.append(int(folder_num.group()))
    run_name = str(max(int_folders + [0]) + 1)+"-"+config.agent+"-"+config.env_name
    config.log_path = "log/" + run_name + "/"
    checkpoint_path = config.log_path + "checkpoint/"
    print("Starting run: " + str(run_name))
    print("On device: "+ str(config.device))
    os.makedirs(checkpoint_path)
    config_log_file = open(config.log_path + "config.txt", 'w+')
    config_vars_dict = vars(config)
    for var in config_vars_dict:
        config_log_file.write(var + ": " + str(config_vars_dict[var]) + "\n")
    config_log_file.close()

agent = Agent(config, sess)

saver = tf.train.Saver(max_to_keep=20)

def load_checkpoint():
    run_num = re.search(r"\d+", config.load_checkpoint)
    if run_num == None:
        raise Exception("Not a run number in checkpoint file")
    run_num = run_num.group()
    for folder in os.listdir("log"):
        folder_num = re.search(r"\d+", folder)
        if folder_num and folder_num.group() == run_num:
            ckpt_file = "log/"+folder+"/checkpoint/"+config.load_checkpoint
    print("loading: " + ckpt_file)
    saver.restore(sess, ckpt_file)

if config.load_checkpoint != "":
    load_checkpoint()
else:
    sess.run(tf.initialize_all_variables())

def test_run(n):
    agent.testing(True)
    score_list = []
    for episode in range(n):
        x, r, done, score = env.reset(), 0, False, 0
        while not done:
            action = agent.step(x, r)
            x, r, done, info = env.step(action)
            score += r
        agent.done()
        score_list.append(score)
    agent.testing(False)
    return score_list

def train():
    for episode in range(config.num_episodes):
        x, r, done, score = env.reset(), 0, False, 0
        ep_begin_t = time.time()
        ep_begin_step_count = agent.step_count
        while not done:
            action = agent.step(x, r)
            x, r, done, info = env.step(action)
            score += r
        agent.done()
        ep_duration = time.time() - ep_begin_t
        if not config.logging:
            continue
        is_final_episode = config.num_episodes == episode
        # online Summary
        if episode % config.log_online_summary_rate == 0 or is_final_episode:
            episode_online_summary = tf.Summary(
                value=[
                    tf.Summary.Value(
                        tag="online/epsilon",
                        simple_value=agent.epsilon()),
                    tf.Summary.Value(
                        tag="score/online",
                        simple_value=score),
                    tf.Summary.Value(
                        tag="online/global_step",
                        simple_value=agent.step_count),
                    tf.Summary.Value(
                        tag="online/step_duration",
                        simple_value=ep_duration/(agent.step_count - ep_begin_step_count)),
                    tf.Summary.Value(
                        tag="online/ep_duration_seconds",
                        simple_value=ep_duration)])
            agent.summary_writter.add_summary(episode_online_summary, episode)
        # log percent
        if episode % config.log_percent_rate == 0 and episode != 0 or is_final_episode:
            percent = float(episode) / config.num_episodes * 100
            print("%i%% -- epsilon:%.2f" % (percent, agent.epsilon()))
        # save
        if episode % config.save_rate == 0 and episode != 0 or is_final_episode:
            print("saving checkpoint at episode " + str(episode))
            saver.save(sess, checkpoint_path + "run-" +
                       run_name + "_episode", episode)
        # performance summary
        if episode % config.log_perf_summary_rate == 0 or is_final_episode:
            for action_mode in agent.action_modes:
                print("testing in mode: "+action_mode)
                agent.set_action_mode(action_mode)
                score_list = test_run(n=config.test_run_num)
                performance_summary = tf.Summary(
                    value=[
                        tf.Summary.Value(
                            tag="score/"+action_mode+"_average",
                            simple_value=sum(score_list)/len(score_list)),
                        tf.Summary.Value(
                            tag="score/"+action_mode+"_max",
                            simple_value=max(score_list)),
                        tf.Summary.Value(
                            tag="score/"+action_mode+"_min",
                            simple_value=min(score_list)),
                    ])
                agent.summary_writter.add_summary(performance_summary, agent.step_count)
            agent.set_action_mode(agent.default_action_mode)

train()
