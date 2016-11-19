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

from ale_python_interface import ALEInterface
import tensorflow as tf
import numpy as np
import cv2
import random
import threading
import sys
import time
import os
from replayMemory import ReplayMemory
from buildGraph import createQNetwork, build_train_op

ale = ALEInterface()
viz = False
rom_name = "roms/Breakout.bin"
ale.setBool('sound', False)
ale.setBool('display_screen', viz)
ale.setInt("frame_skip", 4)
ale.loadROM(rom_name)
legal_actions = ale.getMinimalActionSet()
action_map = {}
for i in range(len(legal_actions)):
    action_map[i] = legal_actions[i]
action_num = len(action_map)

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

if config.h_to_h not in ["oh_concat", "expanded_concat", "conditional"]:
    raise "Not valid transition function"

def get_epsilon():
    if global_step < config.exploration_steps:
        return config.initial_epsilon-((config.initial_epsilon-config.final_epsilon)/config.exploration_steps)*global_step
    else:
        return config.final_epsilon

RM = ReplayMemory(config)

def flush_print(str):
    print(str)
    sys.stdout.flush()

def preprocess(new_frame, state):
    frame = cv2.resize(new_frame, (84, 84))
    new_state = np.roll(state, -1, axis=3)
    new_state[0, :, :, config.buff_size -1] = frame
    return new_state

with tf.device(config.device):
    input_state_ph = tf.placeholder(tf.float32,[config.batch_size,84,84,4], name="input_state_ph")
    # this should be: input_state_placeholder = tf.placeholder("float",[None,84,84,4], name="state_placeholder")
    action_ph = tf.placeholder(tf.int64, [config.batch_size], name="Action_ph")
    Y_ph = tf.placeholder(tf.float32, [config.batch_size], name="Y_ph")
    next_Y_ph = tf.placeholder(tf.float32, [config.batch_size, action_num], name="next_Y_ph")
    reward_ph = tf.placeholder(tf.float32, [config.batch_size], name="reward_ph")

    ph_lst = [input_state_ph, action_ph, Y_ph, next_Y_ph, reward_ph]

    q = tf.FIFOQueue(2, [ph.dtype for ph in ph_lst],
                     [ph.get_shape() for ph in ph_lst])
    enqueue_op = q.enqueue(ph_lst)
    input_state, action, Y, next_Y, reward = q.dequeue()

    # so that i can feed inputs with different batch sizes.
    input_state = tf.placeholder_with_default(input_state, shape=tf.TensorShape([None]).concatenate(input_state.get_shape()[1:]))
    action = tf.placeholder_with_default(action, shape=[None])
    next_input_state_ph = tf.placeholder(tf.float32,[config.batch_size,84,84,4], name="next_input_state_placeholder")

    with tf.variable_scope("DQN"):
        Q, R, predicted_next_Q = createQNetwork(input_state, action, config, "DQN")
        DQN_params = tf.get_collection("DQN_weights")
        max_action_DQN = tf.argmax(Q, 1)
    with tf.variable_scope("DQNTarget"):
        # pasing an action is useless because the target never runs the next_Y_prediction but it is needed for the code to work
        QT, RT, predicted_next_QT = createQNetwork(next_input_state_ph, action, config, "DQNT")
        DQNT_params = tf.get_collection("DQNT_weights")

    # DQN summary
    for i in range(action_num):
        dqni = tf.scalar_summary("DQN/action"+str(i), Q[0, i])
        tf.add_to_collection("DQN_summaries", dqni)

    sync_DQNT_op = [DQNT_params[i].assign(DQN_params[i]) for i in range(len(DQN_params))]

    train_op = build_train_op(Q, Y, R, reward, predicted_next_Q, next_Y, action, config)


_action_ = action
_R_ = R

def enqueue_from_RM():
    while True:
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, _ = RM.sample_transition_batch()
        if global_step % config.save_summary_rate == 0:
            QT_np, DQNT_summary_str = sess.run([QT, DQNT_summary_op], feed_dict={next_input_state_ph:next_state_batch})
            summary_writter.add_summary(DQNT_summary_str, global_step)
        else:
            QT_np = sess.run(QT, feed_dict={next_input_state_ph:next_state_batch})

        DQNT_max_action_batch = np.max(QT_np, 1)
        Y = []
        for i in range(state_batch.shape[0]):
            terminal = terminal_batch[i]
            if terminal:
                Y.append(reward_batch[i])
            else:
                Y.append(reward_batch[i] + config.gamma * DQNT_max_action_batch[i])
        feed_dict={input_state_ph:state_batch, action_ph:action_batch, next_input_state_ph:next_state_batch, Y_ph:Y, next_Y_ph:QT_np, reward_ph:reward_batch}
        sess.run(enqueue_op, feed_dict=feed_dict)

enqueue_from_RM_thread = threading.Thread(target=enqueue_from_RM)
enqueue_from_RM_thread.daemon = True

timeout_option = tf.RunOptions(timeout_in_ms=5000)

def update_params():
    if global_step > config.steps_before_training:
        if enqueue_from_RM_thread.isAlive() == False:
            flush_print("starting enqueue thread")
            enqueue_from_RM_thread.start()

        if global_step % config.save_summary_rate == 0:
            _, DQN_summary_str = sess.run([train_op, DQN_summary_op], options=timeout_option)
            summary_writter.add_summary(DQN_summary_str, global_step)
        else:
             _ = sess.run(train_op, options=timeout_option)

        if global_step % config.sync_rate == 0:
            sess.run(sync_DQNT_op)

sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.log_device_placement = False
sess = tf.Session(config=sess_config)
saver = tf.train.Saver(DQN_params, max_to_keep = 20)
sess.run(tf.initialize_variables(DQN_params))
sess.run(tf.initialize_variables(DQNT_params))
sess.run(tf.initialize_all_variables())

#geneate a new set of paths
run_list = os.listdir("log")
int_run_list = [int(r) for r in run_list] + [0]
run_name = str(max(int_run_list) + 1)
#run_name = str(3)
checkpoint_path = "checkpoint/" + run_name + ".ckpt"
log_path = "log/"+ run_name
print(run_name)
DQN_summary_op = tf.merge_summary(tf.get_collection("DQN_summaries") + \
                                  tf.get_collection("DQN_prediction_summaries"))
DQNT_summary_op = tf.merge_summary(tf.get_collection("DQNT_summaries"))
summary_writter = tf.train.SummaryWriter(log_path, sess.graph, flush_secs=20)

def e_greedy_action(epsilon, state):
        if np.random.uniform() < epsilon:
            action = random.randint(0, action_num - 1)
        else:
            action = np.argmax(sess.run(Q, feed_dict={input_state:state})[0])
        return action

def e_greedy_planning_action(epsilon, state):
        if np.random.uniform() < epsilon:
            a = random.randint(0, action_num - 1)
        else:
            next_Q = []
            predicned_Rs, next_Q_0 = sess.run([_R_, predicted_next_Q], feed_dict={input_state:state, _action_:[0]})
            a = 1
            next_Q.append(np.max(next_Q_0))
            next_Q.append(np.max(sess.run(predicted_next_Q,feed_dict={input_state:state, _action_:[1]})))
            next_Q.append(np.max(sess.run(predicted_next_Q,feed_dict={input_state:state, _action_:[2]})))
            next_Q.append(np.max(sess.run(predicted_next_Q,feed_dict={input_state:state, _action_:[3]})))
            predicted_Q = []
            for i in range(4):
                predicted_Q.append(predicned_Rs[0][i] + config.gamma* next_Q[i])
            a = np.argmax(predicted_Q)
        return a

def greedy_run(epsilon, n, use_planning=False):
    ale.reset_game()
    R_list = []
    for episode in range(n):
        state = np.zeros((1, 84, 84, config.buff_size), dtype=np.uint8)
        state = preprocess(ale.getScreenGrayscale(), state)
        R = 0
        while ale.game_over() == False:
            if use_planning:
                action = e_greedy_planning_action(epsilon, state)
            else:
                action = e_greedy_action(epsilon, state)
            reward = ale.act(action_map[action])
            state = preprocess(ale.getScreenGrayscale(), state)
            R += reward
        R_list.append(R)
        ale.reset_game()
    return R_list

global_step = 0
global_episode = 0
logging = True

t = time.time()
num_episodes = 100000
initial_episode = global_episode
sess.run(sync_DQNT_op)
for episode in range(global_episode, num_episodes + global_episode):
    global state
    state = np.zeros((1, 84, 84, config.buff_size), dtype=np.uint8)
    state = preprocess(ale.getScreenGrayscale(), state)
    R = 0
    ep_begin_t = time.time()
    terminal = False
    pseudo_terminal = False
    lives = ale.lives()
    episode_begining_step = global_step
    while terminal == False:
        action = e_greedy_action(get_epsilon(), state)
        reward = ale.act(action_map[action])
        clipped_reward = max(-1, min(1, reward))
        R += reward
        pseudo_terminal = False
        if ale.game_over():
            terminal = True
        if lives != ale.lives() or terminal:
            lives = ale.lives()
            pseudo_terminal = True
        RM.add(state[0, :, :, config.buff_size -1], action, clipped_reward, pseudo_terminal)
        update_params()
        state = preprocess(ale.getScreenGrayscale(), state)
        global_step += 1
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
    ale.reset_game()
print("==")
print((time.time() - t)/60)
