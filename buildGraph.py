import tensorflow as tf
import numpy as np


def build_activation_summary(x, summaryCollection):
    tensor_name = x.op.name
    hs = tf.histogram_summary(tensor_name + '/activations', x)
    ss = tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.add_to_collection(summaryCollection, hs)
    tf.add_to_collection(summaryCollection, ss)


def conv2d(x, W, stride, name):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID", name=name)


def xavier_std(in_size, out_size):
    return np.sqrt(2. / (in_size + out_size))

#createQNetwork("DQN_summaries", action_num, input_state, action)


def createQNetwork(input_state, action, action_num, summaryCollection):
    conv_layer_counter = [0]
    linear_layer_counter = [0]
    conditional_linear_layer_counter = [0]
    weight_list = []

    def add_conv_layer(nn_head, channels, kernel_size, stride):
        assert len(nn_head.get_shape()
                   ) == 4, "can't add a conv layer to this input"
        conv_layer_counter[0] += 1
        layer_name = "conv" + str(conv_layer_counter[0])
        nn_head_channels = nn_head.get_shape().as_list()[3]
        w_size = [kernel_size, kernel_size, nn_head_channels, channels]

        w = tf.get_variable(layer_name + "_W", w_size, initializer=tf.truncated_normal_initializer(
            stddev=xavier_std(nn_head_channels * kernel_size**2, channels * kernel_size**2)))
        tf.add_to_collection(
            summaryCollection, tf.histogram_summary(w.op.name, w))
        weight_list.append(w)

        new_head = tf.nn.relu(
            conv2d(nn_head, w, stride, name=layer_name), name=layer_name + "_relu")
        build_activation_summary(new_head, summaryCollection)
        return new_head

    def add_linear_layer(nn_head, size, _summaryCollection, layer_name=None, weight_name=None):
        assert len(nn_head.get_shape()
                   ) == 2, "can't add a linear layer to this input"
        if layer_name == None:
            linear_layer_counter[0] += 1
            layer_name = "linear" + str(linear_layer_counter[0])
        if weight_name == None:
            weight_name = layer_name + "_W"
        nn_head_size = nn_head.get_shape().as_list()[1]
        w_size = [nn_head_size, size]

        w = tf.get_variable(weight_name, w_size, initializer=tf.truncated_normal_initializer(
            stddev=xavier_std(nn_head_size, size)))
        if tf.get_variable_scope().reuse == False:
            tf.add_to_collection(_summaryCollection,
                                 tf.histogram_summary(w.op.name, w))
            weight_list.append(w)

        new_head = tf.nn.relu(
            tf.matmul(nn_head, w, name=layer_name), name=layer_name + "_relu")
        build_activation_summary(new_head, _summaryCollection)
        return new_head

    def add_conditional_linear_layer(nn_head, condition, condition_size, size, _summaryCollection, layer_name=None, weight_name=None):
        assert len(nn_head.get_shape()
                   ) == 2, "can't add a linear layer to this input"
        if layer_name == None:
            conditional_linear_layer_counter[0] += 1
            layer_name = "conditional_linear" + \
                str(conditional_linear_layer_counter[0])
        if weight_name == None:
            weight_name = layer_name + "_W"
        nn_head_size = nn_head.get_shape().as_list()[1]
        w_size = [condition_size, nn_head_size, size]

        w = tf.get_variable(weight_name, w_size, initializer=tf.truncated_normal_initializer(
            stddev=xavier_std(nn_head_size, size)))
        if tf.get_variable_scope().reuse == False:
            tf.add_to_collection(_summaryCollection,
                                 tf.histogram_summary(w.op.name, w))
            weight_list.append(w)

        conditional_w = tf.gather(w, condition)
        new_head = tf.nn.relu(
            tf.batch_matmul(tf.expand_dims(nn_head,1), conditional_w, name=layer_name), name=layer_name + "_relu")
        new_head = tf.squeeze(new_head, [1])
        build_activation_summary(new_head, _summaryCollection)
        return new_head

    normalized = input_state / 256.
    tf.add_to_collection(summaryCollection, tf.histogram_summary(
        "normalized_input", normalized))

    nn_head = add_conv_layer(normalized, channels=32, kernel_size=8, stride=4)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=4, stride=2)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=3, stride=1)

    h_conv3_shape = nn_head.get_shape().as_list()
    hidden_state = tf.reshape(
        nn_head, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="conv3_flat")
    hidden_state_shape = hidden_state.get_shape().as_list()

    def hidden_state_to_Q(nn_head, _name, _summaryCollection):
        nn_head = add_linear_layer(nn_head, size=512, _summaryCollection=_summaryCollection,
                                   layer_name="final_linear_" + _name, weight_name="final_linear_Q_W")
        # the last layer is linear without a relu
        nn_head_size = nn_head.get_shape().as_list()[1]
        Q_w = tf.get_variable("Q_W", [nn_head_size, action_num], initializer=tf.truncated_normal_initializer(
            stddev=xavier_std(nn_head_size, action_num)))
        weight_list.append(Q_w)
        Q = tf.matmul(nn_head, Q_w, name=_name)
        tf.add_to_collection(_summaryCollection,
                             tf.histogram_summary(_name, Q))
        return Q

    # the functions state -> Q, and future_state -> future_Q are the same and
    # share parameters
    Q = hidden_state_to_Q(hidden_state, "Q", summaryCollection)

    action_one_hot = tf.one_hot(
        action, action_num, 1., 0., name='action_one_hot')
    predicted_next_hidden_state = tf.concat(
        1, [action_one_hot, hidden_state], name="one_hot_concat_state")
    predicted_next_hidden_state = add_conditional_linear_layer(
        predicted_next_hidden_state, action, action_num, size=512, _summaryCollection=summaryCollection + "_prediction")
    predicted_next_hidden_state = add_conditional_linear_layer(
        predicted_next_hidden_state, action, action_num, size=hidden_state_shape[1], _summaryCollection=summaryCollection + "_prediction")

    tf.get_variable_scope().reuse_variables()
    predicted_next_Q = hidden_state_to_Q(
        predicted_next_hidden_state, "future_Q", summaryCollection + "_prediction")

    return Q, predicted_next_Q, weight_list


def clipped_l2(y, y_t):
    with tf.name_scope("clipped_l2"):
        delta_grad_clip = 1
        batch_delta = y - y_t
        batch_delta_abs = tf.abs(batch_delta)
        batch_delta_quadratic = tf.minimum(batch_delta_abs, delta_grad_clip)
        batch_delta_linear = (batch_delta_abs - batch_delta_quadratic) * 2
        batch = batch_delta_linear + batch_delta_quadratic**2
    return batch

#build_train_op(Q, Y, action, predicted_next_Q, next_Y, action_num, learning_rate)


def build_train_op(Q, Y, action, predicted_next_Q, next_Y, action_num, lr, alpha=1):
    with tf.name_scope("loss"):
        action_one_hot = tf.one_hot(
            action, action_num, 1., 0., name='action_one_hot')
        DQN_acted = tf.reduce_sum(
            Q * action_one_hot, reduction_indices=1, name='DQN_acted')

        loss = tf.reduce_mean(clipped_l2(Y, DQN_acted))
        future_loss = tf.reduce_sum(clipped_l2(
            predicted_next_Q, next_Y), name="future_loss")
        combined_loss = loss + alpha * future_loss

        max_predicted_next_Q = tf.reduce_max(predicted_next_Q, 1)
        next_max_Q = tf.reduce_max(next_Y, 1)
        tf.add_to_collection(
            "DQN_summaries", tf.scalar_summary("rm_average_loss", loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "rm_average_future_loss", future_loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "rm_average_combined_loss", combined_loss))
        tf.add_to_collection(
            "DQN_summaries", tf.scalar_summary("rm_Y_0", Y[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "max_predicted_next_Q_0", max_predicted_next_Q[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "next_max_Q_0", next_max_Q[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "rm_actedDQN_0", DQN_acted[0]))
        #tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_maxDQN_0", tf.reduce_max(DQN[0])))

    opti = tf.train.RMSPropOptimizer(lr, 0.95, 0.95, 0.01)
    grads = opti.compute_gradients(combined_loss)
    train_op = opti.apply_gradients(grads)  # have to pass global_step ?????

    for grad, var in grads:
        if grad is not None:
            tf.add_to_collection("DQN_summaries", tf.histogram_summary(
                var.op.name + '/gradients', grad))

    return train_op
