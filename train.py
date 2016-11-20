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
parser.add_argument("-load_checkpoint", default="")
parser.add_argument("-transition_function", default="oh_concat")
parser.add_argument("-alpha", type=float, default=0.1)
args = parser.parse_args()

import gym
import threading
import sys
import time
import os

#from the selected agent import agent

env = gym.make('breakout-v0')

class config:
    batch_size = args.batch_size
    action_num = action_num
    replay_memory_capacity = args.replay_memory_capacity
    steps_before_training = args.steps_before_training
    buff_size = 4
    device = args.device
    gamma = args.gamma
    learning_rate = args.learning_rate
    exploration_steps = args.exploration_steps
    initial_epsilon = args.initial_epsilon
    final_epsilon = args.final_epsilon
    sync_rate = args.sync_rate
    save_summary_rate = args.save_summary_rate
    alpha = args.alpha
    h_to_h = args.transition_function

# Check that all arguments are valid
if config.h_to_h not in ["oh_concat", "expanded_concat", "conditional"]:
    raise "Not valid transition function"

sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)

saver = tf.train.Saver(DQN_params, max_to_keep = 20)

run_name = str(max([int(r) for r in os.listdir("log")] + [0]) + 1)
log_path = "log/"+ run_name
checkpoint_path = log_path+"checkpoint/"
print("Starting run: " str(run_name))

summary_writter = tf.train.SummaryWriter(log_path, sess.graph, flush_secs=20)

agent = Agent(config, session, summary_writter)

global_step = 0
global_episode = 0
logging = True

def test_run(n):
    score_list = []
    for episode in range(n):
        x, r, done, score = env.reset(), 0, False, 0
        while done:
            action = agent.test_step(x, r)
            x, r, done = env.step(action)
            state = preprocess(ale.getScreenGrayscale(), state)
            score += r
        agent.test_done()
        score_list.append(score)
    return score_list

# Start the training
for episode in range(10):
    x, done = env.reset(), False
    ep_begin_t = time.time()
    while ~done:
        action = agent.step(x, r)
        x, r, done = env.step(action)
        global_step += 1
    agent.done()
    #logs
    ep_duration = time.time() - ep_begin_t
    if logging and episode%100 == 0 and episode != 0 or num_episodes == episode:
        episode_online_summary = tf.Summary(value=[tf.Summary.Value(tag="online/epsilon", simple_value=get_epsilon()),
                                    tf.Summary.Value(tag="online/R", simple_value=R),
                                    tf.Summary.Value(tag="online/steps_in_episode", simple_value= global_step - episode_begining_step),
                                    tf.Summary.Value(tag="online/global_step", simple_value = global_step),
                                    tf.Summary.Value(tag="online/ep_duration_seconds", simple_value=ep_duration)])
        summary_writter.add_summary(episode_online_summary, global_episode)
    # log percent
    if logging and logging==True and episode%500 == 0 and episode != 0 or num_episodes == episode:
        percent = int(float(episode - initial_episode)/num_episodes * 100)
        print("%i%% -- epsilon:%.2f"%(percent, get_epsilon()))
    # save
    if logging and episode%1000 == 0 and episode != 0 or num_episodes == episode:
        print("saving checkpoint at episode " + str(episode))
        saver.save(sess, checkpoint_path, episode)

    # performance summary
    if logging and episode%1000 == 0 and episode != 0 or num_episodes == episode:
        R_list = greedy_run(epsilon = 0.01, n=20)
        Planning_R_list = greedy_run(epsilon = 0.01, n=20, use_planning=True)
        performance_summary = tf.Summary(value=[tf.Summary.Value(tag="R/average", simple_value=sum(R_list)/len(R_list)),
                                      tf.Summary.Value(tag="R/max", simple_value=max(R_list)),
                                      tf.Summary.Value(tag="R/min", simple_value=min(R_list)),
                                      tf.Summary.Value(tag="R/average_planning", simple_value=sum(Planning_R_list)/len(Planning_R_list)),
                                      tf.Summary.Value(tag="R/max_planning", simple_value=max(Planning_R_list)),
                                      tf.Summary.Value(tag="R/min_planning", simple_value=min(Planning_R_list)),
                                      ])
        summary_writter.add_summary(performance_summary, global_step)

    global_episode += 1

# Write summary about the run.
