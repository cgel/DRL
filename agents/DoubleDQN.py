import tensorflow as tf
from DQN import DQN
import commonOps as cops

class DoubleDQN(DQN):

    def __init__(self, config, session):
        DQN.__init__(self, config, session)

    def build_NNs(self):
        with tf.variable_scope("Q") as scope:
            self.Q = self.Q_network(self.state_ph, "DQN")
            scope.reuse_variables()
            # the network with online weights used to select the actions of the target network
            self.DoubleQT = self.Q_network(self.stateT_ph, "DDQNT")

        with tf.variable_scope("QT"):
            self.QT = self.Q_network(
                self.stateT_ph, "DQNT")
            cops.build_scalar_summary(tf.reduce_max(self.QT, 1)[0], "DQNT", "main/next_Q_0")
            cops.build_scalar_summary(tf.reduce_max(self.QT), "DQNT", "main/next_Q_max")


    def Q_target(self):
        target_action = tf.argmax(self.DoubleQT, axis=1)
        target_action_one_hot = tf.one_hot(
            target_action, self.config.action_num, 1., 0., name='target_action_one_hot')

        DoubleQT_acted = tf.reduce_sum(
            self.QT * target_action_one_hot, axis=1, name='DoubleQT')
        Y =  self.reward_ph + self.config.gamma * DoubleQT_acted * (1 - self.terminal_ph)
        return Y
