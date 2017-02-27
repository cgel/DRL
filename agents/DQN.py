import tensorflow as tf
import commonOps as cops
from baseAgent import BaseAgent

# Implementation of DQN form "Human-level control through deep reinforcement learning"

# Note on the usage of collections.
#   For summaries: You might want to write the sumaries of the DQN without also
#                  computing and logging the summaries of the training op and
#                  target network. Therefore they are keep different collections of summaries
#   For variables: to create the sync operation we need a colleciton containing
#                  the variables of the DQN and the target DQN.

class DQN(BaseAgent):

    def __init__(self, config, session):
        BaseAgent.__init__(self, config, session)
        with tf.device(config.device):
            # Create all variables and the FIFOQueue
            self.state_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="state_ph")
            self.stateT_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="stateT_ph")
            self.action_ph = tf.placeholder(tf.int64, [None], name="action_ph")
            self.reward_ph = tf.placeholder(tf.float32, [None], name="reward_ph")
            self.terminal_ph = tf.placeholder(tf.float32, [None], name="terminal_ph")
            self.build_NNs()
            # Note: since the target network is always used together with the train_op
            # they share the colleciton. But if train_op had variables the
            # sync_op constructor will break
            self.train_op = self.train_op("DQNT")
            self.sync_QT_op = []
            for W_pair in zip(
                    tf.get_collection("DQNT_weights"),
                    tf.get_collection("DQN_weights")):
                self.sync_QT_op.append(W_pair[0].assign(W_pair[1]))
            # Define the summary ops
            self.Q_summary_op = tf.merge_summary(
                tf.get_collection("DQN_summaries"))
            self.QT_summary_op = tf.merge_summary(
                tf.get_collection("DQNT_summaries"))

        if config.logging:
                self.summary_writter = tf.train.SummaryWriter(
                    self.config.log_path, self.sess.graph, flush_secs=20)

    # creates the Q network and the target network
    def build_NNs(self):
        with tf.variable_scope("Q"):
            self.Q = self.Q_network(self.state_ph, "DQN")
        with tf.variable_scope("QT"):
            self.QT = self.Q_network(
                self.stateT_ph, "DQNT")
            cops.build_scalar_summary(tf.reduce_max(self.QT, 1)[0], "DQNT", "main/next_Q_0")
            cops.build_scalar_summary(tf.reduce_max(self.QT), "DQNT", "main/next_Q_max")

    def update(self):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, _ = self.RM.sample_transition_batch()
        feed_dict={self.state_ph: state_batch,
                    self.stateT_ph: next_state_batch,
                    self.action_ph: action_batch,
                    self.reward_ph: reward_batch,
                    self.terminal_ph: terminal_batch}
        if self.config.logging and self.step_count % self.config.update_summary_rate == 0:
            _, Q_summary_str, QT_summary_str = self.sess.run(
                [self.train_op, self.Q_summary_op, self.QT_summary_op], feed_dict, options=self.timeout_option)
            self.summary_writter.add_summary(
                Q_summary_str, self.step_count)
            self.summary_writter.add_summary(
                QT_summary_str, self.step_count)
        else:
            _ = self.sess.run(self.train_op, feed_dict, options=self.timeout_option)

        if self.step_count % self.config.sync_rate == 0:
            self.sess.run(self.sync_QT_op)

    # Create the Q network operations
    # If Collection is "DQN", all the variables of this NN are added to the
    # collection "DQN_weights" and all the summaries to the colleciton "DQN_summaries"
    def Q_network(self, input_state, Collection):
        conv_stack_shape=[(32,8,4),
                    (64,4,2),
                    (64,3,1)]
        head = tf.div(input_state, 256., name="normalized_input")
        cops.build_activation_summary(head, Collection)
        head = cops.conv_stack(head, conv_stack_shape, self.config, Collection)
        head = cops.flatten(head)
        head = cops.add_relu_layer(head, size=512, Collection=Collection)
        Q = cops.add_linear_layer(head, self.config.action_num, Collection, layer_name="Q")
        # DQN summary
        for i in range(self.config.action_num):
            cops.build_scalar_summary(Q[0, i], Collection, "Q/Q_0_"+str(i))
        return Q

    def Q_target(self):
        QT_max_action = tf.reduce_max(self.QT, 1)
        return self.reward_ph + self.config.gamma * \
            QT_max_action * (1 - self.terminal_ph)


    def train_op(self, Collection):
        with tf.name_scope("loss"):
            # could be done more efficiently with gather_nd or transpose + gather
            action_one_hot = tf.one_hot(
                self.action_ph, self.config.action_num, 1., 0., name='action_one_hot')
            acted_Q = tf.reduce_sum(
                self.Q * action_one_hot, reduction_indices=1, name='DQN_acted')

            Y = self.Q_target()
            Y = tf.stop_gradient(Y)

            loss_batch = cops.clipped_l2(Y, acted_Q)
            loss = tf.reduce_sum(loss_batch, name="loss")

            cops.build_scalar_summary(loss, Collection, "losses/loss")
            cops.build_scalar_summary(loss_batch[0], Collection, "losses/loss_0")
            cops.build_scalar_summary(tf.reduce_max(loss_batch), Collection, "losses/loss_max")
            cops.build_scalar_summary(Y[0], Collection, "main/Y_0")
            cops.build_scalar_summary(tf.reduce_max(Y), Collection, "main/Y_max")
            cops.build_scalar_summary(acted_Q[0], Collection, "main/acted_Q_0")
            cops.build_scalar_summary(tf.reduce_max(acted_Q), Collection, "main/acted_Q_max")
            cops.build_scalar_summary(tf.reduce_max(self.reward_ph), Collection, "main/reward_max")

        train_op, grads = cops.graves_rmsprop_optimizer(
            loss, self.config.learning_rate, 0.95, 0.01, 1)

        for grad, var in grads:
            if grad is True:
                cops.build_hist_summary(grad, Collection, var.op.name + '/gradients')
        return train_op
