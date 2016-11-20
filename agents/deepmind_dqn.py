import tensorflow as tf
import numpy as np
import cv2
import random
from replayMemory import ReplayMemory
from commonOps import createQNetwork, build_train_op

class Agent:
    def __init__(self, cofig, session, summary_writter):
        # build the net
        self.RM = ReplayMemory(config)
        self.summary_writter = summary_writter

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

            enqueue_from_RM_thread = threading.Thread(target=enqueue_from_RM)
            enqueue_from_RM_thread.daemon = True


            DQN_summary_op = tf.merge_summary(tf.get_collection("DQN_summaries") + \
                                              tf.get_collection("DQN_prediction_summaries"))
            DQNT_summary_op = tf.merge_summary(tf.get_collection("DQNT_summaries"))

            sess.run(tf.initialize_variables(DQN_params))
            sess.run(tf.initialize_variables(DQNT_params))
            sess.run(tf.initialize_all_variables())

        if args.load_checkpoint != "":
            ckpt_file = "checkpoint/" + args.load_checkpoint
            print("loading: " +'"'+args.load_checkpoint+'"')
            saver.restore(sess, ckpt_file)
            sess.run(sync_DQNT_op)

    def step(self, x, r, done):
        action = Q(self.state)
        x,r, done = env.step(action)
        RM.add(self.state, action, r, done)
        self.state = preprocess(x)
        update()

    def step(self, x, r):
        RM.add(self.state, self.action, self.reward, False)
        self.action = self.e_greedy_action(self.epsilon)
        self.state = preprocess(x)
        self.update()
        return self.action

    # Adds the transition to the RM and resets the internal state for the next episode
    def done():
        RM.add(self.state, self.action, self.reward, True)
        #reset state

    def test_step(self, x, r):


    # Resets the internal states for the next test episode
    def test_done():

    timeout_option = tf.RunOptions(timeout_in_ms=5000)

    def update():
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

    def epsilon():
        if global_step < config.exploration_steps:
            return config.initial_epsilon-((config.initial_epsilon-config.final_epsilon)/config.exploration_steps)*global_step
        else:
            return config.final_epsilon

    def preprocess(x, state):
        frame = cv2.resize(new_frame, (84, 84))
        new_state = np.roll(state, -1, axis=3)
        new_state[0, :, :, config.buff_size -1] = frame
        return new_state

    def e_greedy_action(epsilon, state):
            if np.random.uniform() < epsilon:
                action = random.randint(0, action_num - 1)
            else:
                action = np.argmax(sess.run(Q, feed_dict={input_state:state})[0])
            return action

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

    def save(self, save_path):

    def load(self, load_path):
