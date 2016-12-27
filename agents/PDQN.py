import tensorflow as tf
import numpy as np
import commonOps as cops
from base_agent import BaseAgent

class Agent(BaseAgent):

    def __init__(self, config, session):
        BaseAgent.__init__(self, config, session)
        self.action_modes = {str(config.testing_epsilon)+"_greedy":self.e_greedy_action,
                            "plan_"+str(config.testing_epsilon)+"_greedy":self.plan_e_greedy_action}
        self.default_action_mode = self.action_modes.items()[0]
        self.action_mode = self.default_action_mode
        # build the net
        with tf.device(config.device):
            # Create all variables and the FIFOQueue
            self.state_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="state_ph")
            self.action_ph = tf.placeholder(tf.int64, [None], name="action_ph")
            self.reward_ph = tf.placeholder(tf.float32, [None], name="reward_ph")
            self.terminal_ph = tf.placeholder(tf.float32, [None], name="terminal_ph")
            self.stateT_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="stateT_ph")
            # Define all the ops
            with tf.variable_scope("Q"):
                self.h_state = self.state_to_hidden(self.state_ph, config, "Normal")
                self.Q = self.hidden_to_Q(self.h_state, config, "Normal")
                self.predicted_reward = self.hidden_to_reward(self.h_state, config, "Normal")
                self.predicted_h_state = self.hidden_to_hidden(self.h_state, self.action_ph, config, "Normal")
                tf.get_variable_scope().reuse_variables()
                self.predicted_next_Q = self.hidden_to_Q(self.predicted_h_state, config, "Normal")
            with tf.variable_scope("QT"):
                self.h_stateT = self.state_to_hidden(self.stateT_ph, config, "Target")
                self.QT = self.hidden_to_Q(self.h_stateT, config, "Target")

            self.train_op = self.train_op(self.Q, self.predicted_reward,
                                self.predicted_next_Q, self.QT, self.reward_ph,
                                self.action_ph, self.terminal_ph, config, "Normal")
            self.sync_QT_op = []
            for W_pair in zip(
                    tf.get_collection("Target_weights"),
                    tf.get_collection("Normal_weights")):
                self.sync_QT_op.append(W_pair[0].assign(W_pair[1]))
            # Define the summary ops
            self.Q_summary_op = tf.merge_summary(
                tf.get_collection("Normal_summaries"))
            self.QT_summary_op = tf.merge_summary(
                tf.get_collection("Target_summaries"))

        self.summary_writter = tf.train.SummaryWriter(
            self.config.log_path, self.sess.graph, flush_secs=20)


    def update(self):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, _ = self.RM.sample_transition_batch()
        feed_dict={self.state_ph: state_batch,
                    self.stateT_ph: next_state_batch,
                    self.action_ph: action_batch,
                    self.reward_ph: reward_batch,
                    self.terminal_ph: terminal_batch}
        if self.config.logging and self.step_count % self.config.update_summary_rate == 0:
            _, Q_summary_str = self.sess.run(
                [self.train_op, self.Q_summary_op], feed_dict, options=self.timeout_option)
            self.summary_writter.add_summary(
                Q_summary_str, self.step_count)
        else:
            _ = self.sess.run(self.train_op, feed_dict, options=self.timeout_option)

        if self.step_count % self.config.sync_rate == 0:
            self.sess.run(self.sync_QT_op)

    def state_to_hidden(self, input_state, config, Collection=None):
        conv_stack_shape=[(32,8,4),
                    (64,4,2),
                    (64,3,1)]
        head = tf.div(input_state, 256., name="normalized_input")
        cops.build_activation_summary(head, Collection)
        head = cops.conv_stack(head, conv_stack_shape, config, Collection)
        head = cops.flatten(head)
        return head

        # will be called twice under the same var scope
    def hidden_to_Q(self, head, config, Collection):
        suffix = ""
        if tf.get_variable_scope().reuse:
            suffix = "_prediction"
        head = cops.add_relu_layer(head, size=512, Collection=Collection,
                              layer_name="final_relu_layer" + suffix, weight_name="final_linear_Q_W")
        Q = cops.add_linear_layer(head, config.action_num, Collection, layer_name="Q"+suffix, weight_name="Q_W")
        for i in range(config.action_num):
            tf.scalar_summary("DQN/action"+suffix +"_"+str(i),
                                     Q[0, i], collections=["Q_summaries"])
        return Q


    def hidden_to_reward(self, head, config, Collection):
        head = cops.add_relu_layer(head, size=256, layer_name="r_relu1", Collection=Collection)
        # the last layer is linear without a relu
        r = cops.add_linear_layer(head, config.action_num, Collection, layer_name="r_linear2", weight_name="r_W")
        tf.add_to_collection(Collection + "_summaries",
                             tf.histogram_summary("r", r))
        return r


    def hidden_to_hidden(self, head, action, config, Collection):
        hidden_state_shape = head.get_shape().as_list()
        action_one_hot = tf.one_hot(
            action, config.action_num, 1., 0., name='action_one_hot')
        head = tf.concat(
            1, [action_one_hot, head], name="one_hot_concat_state")
        head = cops.add_relu_layer(
            head, size=256, layer_name="prediction_relu1", Collection=Collection)
        head = cops.add_relu_layer(
            head, size=hidden_state_shape[1], layer_name="prediction_hidden", Collection=Collection)
        return head

    def train_op(self, Q, predicted_reward, predicted_next_Q, QT, reward, action, terminal, config, Collection):
        QT = tf.stop_gradient(QT)
        with tf.name_scope("loss"):
            # could be done more efficiently with gather_nd or transpose + gather
            action_one_hot = tf.one_hot(
                action, config.action_num, 1., 0., name='action_one_hot')
            acted_Q = tf.reduce_sum(
                Q * action_one_hot, reduction_indices=1, name='DQN_acted')
            predicted_reward_action = tf.reduce_sum(
                predicted_reward * action_one_hot, reduction_indices=1, name='DQN_acted')

            QT_max_action = tf.reduce_max(QT, 1)
            Y = reward + config.gamma * \
                QT_max_action * (1 - terminal)

            Q_loss = tf.reduce_sum(
                cops.clipped_l2(Y, acted_Q), name="Q_loss")

            # note that the target is defined over all actions
            predicted_Q_loss = config.alpha / config.action_num * tf.reduce_sum(cops.clipped_l2(
                predicted_next_Q, QT, grad_clip=config.alpha), name="future_loss")

            predicted_reward_loss = config.alpha * tf.reduce_sum(
                cops.clipped_l2(predicted_reward_action, reward), name="R_loss")

            # maybe add the linear factor 2(DQNR-real_R)(predicted_next_Q-next_Y)
            combined_loss = Q_loss + predicted_Q_loss + predicted_reward_loss

            tf.scalar_summary(
                "losses/Q", Q_loss, collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "losses/predicted_Q", predicted_Q_loss, collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "losses/predicted_reward", predicted_reward_loss, collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "losses/combined", combined_loss, collections=[Collection + "_summaries"])

            tf.scalar_summary(
                "main/Y_0", Y[0], collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/Y_max", tf.reduce_max(Y), collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/QT_max_action_0", QT_max_action[0], collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/acted_Q_0", acted_Q[0], collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/acted_Q_max", tf.reduce_max(acted_Q), collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/max_predicted_Q_0", tf.reduce_max(predicted_next_Q, 1)[0], collections=[Collection + "_summaries"])

            tf.scalar_summary(
                "main/predicted_reward_0", predicted_reward_action[0], collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/reward_max", tf.reduce_max(reward), collections=[Collection + "_summaries"])

        train_op, grads = cops.graves_rmsprop_optimizer(
            combined_loss, config.learning_rate, 0.95, 0.01, 1)

        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad, name=var.op.name +
                                     '/gradients', collections=[Collection + "_summaries"])
        return train_op

    def plan_e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        else:
            #should be optimized
            # instead of dequeuing feed the game state
            feed_dict={self.state_ph:self.game_state, self.action_ph:[0]} # action 0
            predicted_Rs, predicted_next_Q_0 = self.sess.run([self.predicted_reward, self.predicted_next_Q], feed_dict)
            predicted_next_Qs = np.zeros(self.config.action_num)
            predicted_next_Qs[0] = np.max(predicted_next_Q_0)
            for a in range(1, self.config.action_num):
                feed_dict={self.state_ph:self.game_state, self.action_ph:[a]}
                predicted_next_Qs[a] == np.max(self.sess.run(self.predicted_next_Q, feed_dict))
            action = np.argmax(predicted_Rs + self.config.gamma * predicted_next_Qs)
        return action
