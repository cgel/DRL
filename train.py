import tensorflow as tf
import gym
import time
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-replay_memory_capacity", type=int, default=1000000)
parser.add_argument("-steps_before_training", type=int, default=12500)
parser.add_argument("-exploration_steps", type=int, default=250000)
parser.add_argument("-sync_rate", type=int, default=2500)
parser.add_argument("-save_summary_rate", type=int, default=1000)
parser.add_argument("-device", default="/gpu:0")
parser.add_argument("-gamma", type=float, default=0.99)
parser.add_argument("-learning_rate", type=float, default=0.00025)
parser.add_argument("-initial_epsilon", type=float, default=1.)
parser.add_argument("-final_epsilon", type=float, default=0.1)
parser.add_argument("-buff_size", type=float, default=4)
parser.add_argument("-load_checkpoint", default="")
parser.add_argument("-logging", default="")
parser.add_argument("-transition_function", default="oh_concat")
parser.add_argument("-alpha", type=float, default=0.1)
config = parser.parse_args()

config.num_episodes = 10000
config.logging = config.logging not in ["0", "false", "False"]
if config.transition_function not in ["oh_concat", "expanded_concat", "conditional"]:
    raise "Not valid transition function"

# from the selected agent import agent
from agents.deepmind_dqn import Agent

env = gym.make('Breakout-v0')
config.action_num = env.action_space.n

sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)

if config.logging:
    run_name = str(max([int(r) for r in os.listdir("log")] + [0]) + 1)
    log_path = "log/" + run_name
    checkpoint_path = log_path + "checkpoint/"
    print("Starting run: "+str(run_name))

summary_writter = tf.train.SummaryWriter(log_path, sess.graph, flush_secs=20)

agent = Agent(config, sess, summary_writter)

saver = tf.train.Saver(sess, max_to_keep=20)
if config.load_checkpoint != "":
    ckpt_file = "checkpoint/" + config.load_checkpoint
    print("loading: " + '"' + config.load_checkpoint + '"')
    saver.restore(sess, ckpt_file)
else:
    sess.run(tf.initialize_all_variables())

def test_run(n):
    agent.testing()
    score_list = []
    for episode in range(n):
        x, r, done, score = env.reset(), 0, False, 0
        while done:
            action = agent.step(x, r)
            x, r, done = env.step(action)
            score += r
        agent.done()
        score_list.append(score)
    agent.training()
    return score_list

# Start the training
for episode in range(config.num_episodes):
    x, r, done, score = env.reset(), 0, False, 0
    ep_begin_t = time.time()
    while ~done:
        action = agent.step(x, r)
        x, r, done = env.step(action)
        score += r
    agent.done()
    ep_duration = time.time() - ep_begin_t
    if config.logging and episode % 100 == 0 and episode != 0 or config.num_episodes == episode:
        episode_online_summary = tf.Summary(value=[tf.Summary.Value(tag="online/epsilon", simple_value=agent.epsilon()),
                                                   tf.Summary.Value(
                                                       tag="online/R", simple_value=score),
                                                   tf.Summary.Value(
                                                       tag="online/global_step", simple_value=agent.step),
                                                   tf.Summary.Value(tag="online/ep_duration_seconds", simple_value=ep_duration)])
        summary_writter.add_summary(episode_online_summary, episode)
    # log percent
    if config.logging and episode % 500 == 0 and episode != 0 or config.num_episodes == episode:
        percent = float(episode) / config.num_episodes * 100
        print("%i%% -- epsilon:%.2f" % (percent, agent.epsilon()))
    # save
    if config.logging and episode % 1000 == 0 and episode != 0 or config.num_episodes == episode:
        print("saving checkpoint at episode " + str(episode))
        saver.save(sess, checkpoint_path, episode)

    # performance summary
    if config.logging and episode % 1000 == 0 and episode != 0 or config.num_episodes == episode:
        R_list = test_run(n=20)
        performance_summary = tf.Summary(value=[tf.Summary.Value(tag="R/average", simple_value=sum(R_list) / len(R_list)),
                                                tf.Summary.Value(
                                                    tag="R/max", simple_value=max(R_list)),
                                                tf.Summary.Value(
                                                    tag="R/min", simple_value=min(R_list)),
                                                ])
        summary_writter.add_summary(performance_summary, agent.step)

# Write summary about the run.
