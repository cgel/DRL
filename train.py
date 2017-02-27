import tensorflow as tf
import time
import numpy as np
import parseConfig
import utils
import importlib

config = parseConfig.config

env = utils.create_env(config)

tf.logging.set_verbosity(tf.logging.ERROR)
sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)

Agent = getattr(importlib.import_module("agents."+config.agent), config.agent)
agent = Agent(config, sess)

saver = tf.train.Saver(max_to_keep=20)

if config.load_checkpoint != "":
    utils.load_checkpoint()
else:
    sess.run(tf.initialize_all_variables())

print("Starting run: " + str(config.run_name))
print("Using agent "+config.agent)
print("On device: "+ config.device)

def test_run(n):
    agent.testing(True)
    score_list = []
    for episode in range(n):
        x, r, done, score = env.reset(), 0, False, 0
        while not done:
            action = agent.step(x, r)
            x, r, done, info = env.step(action)
            score += r
        agent.terminal()
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
        agent.terminal()
        ep_duration = time.time() - ep_begin_t
        if not config.logging:
            continue
        is_final_episode = config.num_episodes == episode
        if episode % config.log_online_summary_rate == 0 or is_final_episode:
            online_summary(episode, score, ep_duration, agent.step_count - ep_begin_step_count)
        # log to console
        if episode % config.log_console_rate == 0 and episode != 0 or is_final_episode:
            percent = float(episode) / config.num_episodes * 100
            print("%i%% -- %s %s" % (percent, config.run_name, config.device))
        # save
        if episode % config.save_rate == 0 and episode != 0 or is_final_episode:
            print("saving checkpoint at episode " + str(episode))
            saver.save(sess, config.checkpoint_path + "run-" +
                       config.run_name + "_episode", episode)
        if episode % config.log_perf_summary_rate == 0 or is_final_episode:
            performance_summary()


def online_summary(ep_num, score, ep_duration, ep_step_count):
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
                simple_value=ep_duration/ep_step_count),
            tf.Summary.Value(
                tag="online/ep_step_count",
                simple_value=ep_step_count),
            tf.Summary.Value(
                tag="online/ep_duration_seconds",
                simple_value=ep_duration)])
    agent.summary_writter.add_summary(episode_online_summary, ep_num)

def performance_summary():
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
