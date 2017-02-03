import tensorflow as tf
from DQN import DQN

class DoubleDQN(DQN):

    def __init__(self, config, session):
        DQN.__init__(self, config, session)

    def build_NNs(self):
        with tf.variable_scope("Q") as scope:
            self.Q = self.Q_network(self.state_ph, "Normal")
            scope.reuse_variables()
            # the network with online weights used to select the actions of the target network
            self.DoubleQ = self.Q_network(self.stateT_ph, "Target")

        with tf.variable_scope("QT"):
            self.QT = self.Q_network(
                self.stateT_ph, "Target")
            tf.scalar_summary(
                "main/next_Q_max", tf.reduce_max(self.QT), collections=["Target_summaries"])
            tf.scalar_summary(
                "main/next_Q_0", tf.reduce_max(self.QT, 1)[0], collections=["Target_summaries"])

    def Q_target(self):
        target_action = tf.argmax(self.DoubleQ, axis=1)
        target_action_one_hot = tf.one_hot(
            target_action, self.config.action_num, 1., 0., name='target_action_one_hot')
        DoubleQT_acted = tf.reduce_sum(
            self.Q * target_action_one_hot, reduction_indices=1, name='QT_acted')
        return self.reward_ph + self.config.gamma * \
            DoubleQT_acted * (1 - self.terminal_ph)
