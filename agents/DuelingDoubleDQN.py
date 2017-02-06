import tensorflow as tf
import commonOps as cops
from DoubleDQN import DoubleDQN


class DuelingDoubleDQN(DoubleDQN):

    def __init__(self, config, session):
        #just initialize DQN with a different network
        DoubleDQN.__init__(self, config, session)

    def Q_network(self, input_state, Collection=None):
        conv_stack_shape=[(32,8,4),
                    (64,4,2),
                    (64,3,1)]
        head = tf.div(input_state, 256., name="normalized_input")
        cops.build_activation_summary(head, Collection)
        head = cops.conv_stack(head, conv_stack_shape, self.config, Collection)
        head = cops.flatten(head)
        V_head = cops.add_relu_layer(head, size=512, Collection=Collection)
        V = cops.add_linear_layer(V_head, 1, Collection, layer_name="V")
        A_head = cops.add_relu_layer(head, size=512, Collection=Collection)
        A = cops.add_linear_layer(A_head, self.config.action_num, Collection, layer_name="A")
        Q = tf.add(A, V - tf.expand_dims(tf.reduce_mean(A, axis=1)/self.config.action_num, axis=1) )
        # DQN summary
        tf.scalar_summary("Q/V_0_",
                                 V[0], collections=["Normal"])
        for i in range(self.config.action_num):
            tf.scalar_summary("Q/Q_0_" + str(i),
                                     Q[0, i], collections=["Normal"])
            tf.scalar_summary("Q/A_0_" + str(i),
                                     A[0, i], collections=["Normal"])
        return Q
