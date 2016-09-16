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

def add_conv_layer(nn_head, channels, kernel_size, stride, summaryCollection):
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

def add_linear_layer(nn_head, size, summaryCollection, layer_name=None, weight_name=None):
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
        tf.add_to_collection(summaryCollection,
                             tf.histogram_summary(w.op.name, w))
        weight_list.append(w)

    new_head = tf.nn.relu(
        tf.matmul(nn_head, w, name=layer_name), name=layer_name + "_relu")
    build_activation_summary(new_head, summaryCollection)
    return new_head

def add_conditional_linear_layer(nn_head, condition, condition_size, size, summaryCollection, layer_name=None, weight_name=None):
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
        tf.add_to_collection(summaryCollection,
                             tf.histogram_summary(w.op.name, w))
        weight_list.append(w)

    conditional_w = tf.gather(w, condition)
    new_head = tf.nn.relu(
        tf.batch_matmul(tf.expand_dims(nn_head,1), conditional_w, name=layer_name), name=layer_name + "_relu")
    new_head = tf.squeeze(new_head, [1])
    build_activation_summary(new_head, summaryCollection)
    return new_head

conv_layer_counter = [0]
linear_layer_counter = [0]
conditional_linear_layer_counter = [0]
weight_list = []

def hidden_state_to_Q(hidden_state, _name, action_num, summaryCollection):
    nn_head = add_linear_layer(hidden_state, size=512, summaryCollection=summaryCollection,
                               layer_name="final_linear_" + _name, weight_name="final_linear_Q_W")
    # the last layer is linear without a relu
    nn_head_size = nn_head.get_shape().as_list()[1]
    Q_w = tf.get_variable("Q_W", [nn_head_size, action_num], initializer=tf.truncated_normal_initializer(
        stddev=xavier_std(nn_head_size, action_num)))
    weight_list.append(Q_w)
    Q = tf.matmul(nn_head, Q_w, name=_name)
    tf.add_to_collection(summaryCollection,
                         tf.histogram_summary(_name, Q))
    return Q

def hidden_state_to_R(hidden_state, _name, action_num, summaryCollection):
    nn_head = add_linear_layer(hidden_state, size=256, summaryCollection=summaryCollection,
                               layer_name="linear1_" + _name, weight_name="linear_R_W")
    # the last layer is linear without a relu
    nn_head_size = nn_head.get_shape().as_list()[1]
    R_w = tf.get_variable("R_W", [nn_head_size, action_num], initializer=tf.truncated_normal_initializer(
        stddev=xavier_std(nn_head_size, action_num)))
    weight_list.append(R_w)
    R = tf.matmul(nn_head, R_w, name=_name)
    tf.add_to_collection(summaryCollection,
                         tf.histogram_summary(_name, R))
    return R

def hidden_state_to_next_hidden_state_concat(hidden_state, action, action_num, summaryCollection):
    hidden_state_shape = hidden_state.get_shape().as_list()
    action_one_hot = tf.one_hot(
        action, action_num, 1., 0., name='action_one_hot')
    predicted_next_hidden_state = tf.concat(
        1, [action_one_hot, hidden_state], name="one_hot_concat_state")
    predicted_next_hidden_state = add_linear_layer(
        hidden_state, size=256, layer_name="prediction_linear1", summaryCollection=summaryCollection)
    predicted_next_hidden_state = add_linear_layer(
        predicted_next_hidden_state, size=hidden_state_shape[1], layer_name="prediction_linear2", summaryCollection=summaryCollection)
    return predicted_next_hidden_state

def hidden_state_to_next_hidden_state_gather(hidden_state, action, action_num, summaryCollection):
    hidden_state_shape = hidden_state.get_shape().as_list()
    action_one_hot = tf.one_hot(
        action, action_num, 1., 0., name='action_one_hot')
    predicted_next_hidden_state = tf.concat(
        1, [action_one_hot, hidden_state], name="one_hot_concat_state")
    predicted_next_hidden_state = add_linear_layer(
        hidden_state, size=256, layer_name="prediction_linear1", summaryCollection=summaryCollection)
    predicted_next_hidden_state = add_linear_layer(
        predicted_next_hidden_state, size=hidden_state_shape[1], layer_name="prediction_linear2", summaryCollection=summaryCollection)
    return predicted_next_hidden_state

def createQNetwork(input_state, action, action_num, summaryCollection=None):
    normalized = input_state / 256.
    tf.add_to_collection(summaryCollection, tf.histogram_summary(
        "normalized_input", normalized))

    nn_head = add_conv_layer(normalized, channels=32, kernel_size=8, stride=4, summaryCollection=summaryCollection)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=4, stride=2, summaryCollection=summaryCollection)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=3, stride=1, summaryCollection=summaryCollection)

    h_conv3_shape = nn_head.get_shape().as_list()
    nn_head = tf.reshape(
        nn_head, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="conv3_flat")

    hidden_state = add_linear_layer(
        nn_head, size=256, summaryCollection=summaryCollection)

    # the functions state -> Q, and future_state -> future_Q are the same and
    # share parameters
    Q = hidden_state_to_Q(hidden_state, "Q", action_num, summaryCollection)
    R = hidden_state_to_R(hidden_state, "R", action_num, summaryCollection)

    predicted_next_hidden_state = hidden_state_to_next_hidden_state(hidden_state, action, action_num, summaryCollection+"_prediction")

    tf.get_variable_scope().reuse_variables()
    predicted_next_Q = hidden_state_to_Q(
        predicted_next_hidden_state, "future_Q", action_num, summaryCollection + "_prediction")

    return Q, R, predicted_next_Q, weight_list


def clipped_l2(y, y_t, grad_clip=1):
    with tf.name_scope("clipped_l2"):
        batch_delta = y - y_t
        batch_delta_abs = tf.abs(batch_delta)
        batch_delta_quadratic = tf.minimum(batch_delta_abs, grad_clip)
        batch_delta_linear = (
            batch_delta_abs - batch_delta_quadratic) * 2 * grad_clip
        batch = batch_delta_linear + batch_delta_quadratic**2
    return batch


def build_train_op(Q, Y, DQNR, real_R, predicted_next_Q, next_Y, action, action_num, lr, gamma, alpha):
    with tf.name_scope("loss"):
        # could be done more efficiently with gather_nd or transpose + gather
        action_one_hot = tf.one_hot(
            action, action_num, 1., 0., name='action_one_hot')
        DQN_acted = tf.reduce_sum(
            Q * action_one_hot, reduction_indices=1, name='DQN_acted')
        DQNR_acted = tf.reduce_sum(
            DQNR * action_one_hot, reduction_indices=1, name='DQN_acted')

        Q_loss = tf.reduce_mean(clipped_l2(Y, DQN_acted))
        future_loss = alpha * tf.reduce_sum(clipped_l2(
            predicted_next_Q, next_Y, grad_clip=alpha), name="future_loss")
        R_loss = tf.reduce_mean(clipped_l2(DQNR_acted, real_R))

        # maybe add the linear factor 2(DQNR-real_R)(predicted_next_Q-next_Y)
        combined_loss = Q_loss + future_loss + R_loss

        max_predicted_next_Q_0 = tf.reduce_max(predicted_next_Q, 1)[0]
        next_max_Y = tf.reduce_max(next_Y, 1)
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "losses/Q", Q_loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "losses/next_Q", future_loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "losses/R", R_loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "losses/combined", combined_loss))

        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/Y_0", Y[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/acted_Q_0", DQN_acted[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/acted_Q_prediction_0", DQNR_acted[0] * gamma * max_predicted_next_Q_0))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/max_predicted_next_Q_0", max_predicted_next_Q_0))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/next_max_Y_0", next_max_Y[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary(
            "main/R_0", DQNR_acted[0]))

    opti = tf.train.RMSPropOptimizer(lr, 0.95, 0.95, 0.01)
    grads = opti.compute_gradients(combined_loss)
    train_op = opti.apply_gradients(grads)  # have to pass global_step ?????

    for grad, var in grads:
        if grad is not None:
            tf.add_to_collection("DQN_summaries", tf.histogram_summary(
                var.op.name + '/gradients', grad, name=var.op.name + '/gradients'))

    return train_op
