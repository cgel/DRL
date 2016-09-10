import tensorflow as tf
import numpy as np

def build_activation_summary(x, summaryCollection):
    tensor_name = x.op.name
    hs = tf.histogram_summary(tensor_name + '/activations', x)
    ss = tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.add_to_collection(summaryCollection, hs)
    tf.add_to_collection(summaryCollection, ss)

def conv2d(x, W, stride, name):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID", name=name)

def xavier_std(in_size, out_size):
    return np.sqrt(2./(in_size + out_size))

def createQNetwork(summaryCollection, action_num, action_placeholder):
    conv_layer_counter = [0]
    linear_layer_counter = [0]
    conditional_linear_layer_counter = [0]
    weight_list = []

    def add_conv_layer(nn_head, channels, kernel_size, stride):
        assert len(nn_head.get_shape()) == 4, "can't add a conv layer to this input"
        conv_layer_counter[0] += 1
        layer_name = "conv"+str(conv_layer_counter[0])
        nn_head_channels = nn_head.get_shape().as_list()[3]
        w_size = [kernel_size, kernel_size, nn_head_channels, channels]

        w = tf.get_variable(layer_name+"_W", w_size, initializer=tf.truncated_normal_initializer(stddev = xavier_std(nn_head_channels * kernel_size**2, channels * kernel_size**2)))
        tf.add_to_collection(summaryCollection, tf.histogram_summary(w.op.name, w))
        weight_list.append(w)

        new_head = tf.nn.relu(conv2d(nn_head, w, stride, name=layer_name), name=layer_name+"_relu")
        build_activation_summary(new_head, summaryCollection)
        return new_head

    def add_linear_layer(nn_head, size, _summaryCollection, layer_name=None, weight_name=None):
        assert len(nn_head.get_shape()) == 2, "can't add a linear layer to this input"
        if layer_name == None:
            linear_layer_counter[0] +=1
            layer_name = "linear"+str(linear_layer_counter[0])
        if weight_name == None:
            weight_name = layer_name+"_W"
        nn_head_size = nn_head.get_shape().as_list()[1]
        w_size = [nn_head_size, size]

        w = tf.get_variable(weight_name, w_size, initializer=tf.truncated_normal_initializer(stddev = xavier_std(nn_head_size, size)))
        if tf.get_variable_scope().reuse == False:
            tf.add_to_collection(_summaryCollection, tf.histogram_summary(w.op.name, w))
            weight_list.append(w)

        new_head = tf.nn.relu(tf.matmul(nn_head, w, name=layer_name), name=layer_name+"_relu")
        build_activation_summary(new_head, _summaryCollection)
        return new_head

    input_state_placeholder = tf.placeholder("float",[None,84,84,4], name=summaryCollection+"/state_placeholder")
    # this should be: input_state_placeholder = tf.placeholder("float",[None,84,84,4], name="state_placeholder")
    normalized = input_state_placeholder / 256.
    tf.add_to_collection(summaryCollection, tf.histogram_summary("normalized_input", normalized))

    nn_head = add_conv_layer(normalized, channels=32, kernel_size=8, stride=4)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=4, stride=2)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=3, stride=1)

    h_conv3_shape = nn_head.get_shape().as_list()
    state = tf.reshape(nn_head, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="conv3_flat")
    state_shape = state.get_shape().as_list()

    def state_to_Q(nn_head, _name, _summaryCollection):
        nn_head = add_linear_layer(nn_head, size=512, _summaryCollection=_summaryCollection, layer_name="final_linear_"+_name, weight_name="final_linear_Q_W")
        # the last layer is linear without a relu
        nn_head_size = nn_head.get_shape().as_list()[1]
        Q_w = tf.get_variable("Q_W", [nn_head_size, action_num], initializer=tf.truncated_normal_initializer(stddev = xavier_std(nn_head_size, action_num)))
        weight_list.append(Q_w)
        Q = tf.matmul(nn_head, Q_w, name=_name)
        tf.add_to_collection(_summaryCollection, tf.histogram_summary(_name, Q))
        return Q

    # the functions state -> Q, and future_state -> future_Q are the same and share parameters
    Q = state_to_Q(state, "Q", summaryCollection)

    # lets take this layer as the state for next state predictions
    #append one hot condition to head_future
    action_one_hot = tf.one_hot(action_placeholder, action_num, 1., 0., name='action_one_hot')
    nn_head_future = tf.concat(1, [action_one_hot, state], name="one_hot_concat_state")
    nn_head_future = add_linear_layer(nn_head_future, size=512, _summaryCollection=summaryCollection+"_prediction", layer_name="future_state_linear_1")
    future_state = add_linear_layer(nn_head_future, size=state_shape[1], _summaryCollection=summaryCollection+"_prediction", layer_name="future_state_linear_2")

    tf.get_variable_scope().reuse_variables()
    future_Q = state_to_Q(future_state, "future_Q", summaryCollection+"_prediction")

    return input_state_placeholder, Q, future_Q, weight_list

def clipped_l2(y, y_t):
    with tf.name_scope("clipped_l2"):
        delta_grad_clip = 1
        batch_delta = y - y_t
        batch_delta_abs = tf.abs(batch_delta)
        batch_delta_quadratic = tf.minimum(batch_delta_abs, delta_grad_clip)
        batch_delta_linear = (batch_delta_abs - batch_delta_quadratic)*2
        batch = batch_delta_linear + batch_delta_quadratic**2
    return batch

def build_train_op(DQN, Y, action, future_Q, next_max_Q, action_num, lr, alpha=1):
    with tf.name_scope("loss"):
        action_one_hot = tf.one_hot(action, action_num, 1., 0., name='action_one_hot')
        DQN_acted = tf.reduce_sum(DQN * action_one_hot, reduction_indices=1, name='DQN_acted')
        max_future_Q = tf.reduce_max(future_Q, 1)

        loss = tf.reduce_mean(clipped_l2(Y, DQN_acted))
        future_loss = tf.reduce_sum(clipped_l2(max_future_Q, next_max_Q), name="future_loss")
        combined_loss = loss + alpha*future_loss

        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_average_loss", loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_average_future_loss", future_loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_average_combined_loss", combined_loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_Y_0", Y[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("max_future_Q_0", max_future_Q[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("next_max_Q_0", next_max_Q[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_actedDQN_0", DQN_acted[0]))
        #tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_maxDQN_0", tf.reduce_max(DQN[0])))

    opti = tf.train.RMSPropOptimizer(lr,0.95,0.95,0.01)
    #opti = tf.train.GradientDescentOptimizer(lr)
    grads = opti.compute_gradients(combined_loss)

    train_op = opti.apply_gradients(grads) # have to pass global_step ?????

    for grad, var in grads:
        if grad is not None:
            tf.add_to_collection("DQN_summaries", tf.histogram_summary(var.op.name + '/gradients', grad))

    return train_op
