import tensorflow as tf
import numpy as np

relu_layer_counter = [0]
conv_layer_counter = [0]
linear_layer_counter = [0]
conditional_linear_layer_counter = [0]

def deepmind_Q(input_state, config, Collection=None):
    normalized = tf.div(input_state, 256., name="normalized_input")
    build_activation_summary(normalized, Collection)

    head = add_conv_layer(normalized, channels=32,
                          kernel_size=8, stride=4, Collection=Collection)
    head = add_conv_layer(head, channels=64,
                          kernel_size=4, stride=2, Collection=Collection)
    head = add_conv_layer(head, channels=64,
                          kernel_size=3, stride=1, Collection=Collection)

    h_conv3_shape = head.get_shape().as_list()
    head = tf.reshape(
        #head, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="flat")
        head, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="conv3_flat")

    head = add_relu_layer(head, size=512, Collection=Collection)
    # the last layer is linear without a relu
    Q = add_linear_layer(head, config.action_num, Collection, layer_name="Q")

    # DQN summary
    for i in range(config.action_num):
        tf.scalar_summary("DQN/action" + str(i),
                                 Q[0, i], collections=["Q_summaries"])
    return Q


def state_to_hidden(input_state, config, Collection=None):
    normalized = input_state / 256.
    tf.add_to_collection(Collection + "_summaries", tf.histogram_summary(
        "normalized_input", normalized))

    head = add_conv_layer(normalized, channels=32,
                          kernel_size=8, stride=4, Collection=Collection)
    head = add_conv_layer(head, channels=64,
                          kernel_size=4, stride=2, Collection=Collection)
    head = add_conv_layer(head, channels=64,
                          kernel_size=3, stride=1, Collection=Collection)

    head_shape = head.get_shape().as_list()
    head = tf.reshape(
        head, [-1, head_shape[1] * head_shape[2] * head_shape[3]], name=head.op.name + "_flat")
    return head

# will be called twice under the same var scope
def hidden_to_Q(hidden_state, config, Collection):
    suffix = ""
    if tf.get_variable_scope().reuse:
        suffix = "_prediction"
    head = add_relu_layer(hidden_state, size=512, Collection=Collection,
                          layer_name="final_relu_layer" + suffix, weight_name="final_linear_Q_W")
    # the last layer is linear without a relu
    Q = add_linear_layer(head, config.action_num, Collection, layer_name="Q"+suffix, weight_name="Q_W")
    tf.add_to_collection(Collection + "_summaries",
                         tf.histogram_summary("Q"+suffix, Q))
    return Q


def hidden_to_r(hidden_state, config, Collection):
    head = add_relu_layer(hidden_state, size=256, layer_name="r_relu1", Collection=Collection)
    # the last layer is linear without a relu
    r = add_linear_layer(head, config.action_num, Collection, layer_name="r_linear2", weight_name="r_W")
    tf.add_to_collection(Collection + "_summaries",
                         tf.histogram_summary("r", r))
    return r


def hidden_to_hidden(hidden_state, action, config, Collection):
    hidden_state_shape = hidden_state.get_shape().as_list()
    action_one_hot = tf.one_hot(
        action, config.action_num, 1., 0., name='action_one_hot')
    head = tf.concat(
        1, [action_one_hot, hidden_state], name="one_hot_concat_state")
    head = add_relu_layer(
        head, size=256, layer_name="prediction_relu1", Collection=Collection)
    head = add_relu_layer(
        head, size=hidden_state_shape[1], layer_name="prediction_hidden", Collection=Collection)
    return head

# -- Training ops --
def dqn_train_op(Q, QT, action, reward, terminal, config, Collection):
    with tf.name_scope("loss"):
        # could be done more efficiently with gather_nd or transpose + gather
        action_one_hot = tf.one_hot(
            action, config.action_num, 1., 0., name='action_one_hot')
        acted_Q = tf.reduce_sum(
            Q * action_one_hot, reduction_indices=1, name='DQN_acted')

        QT_max_action = tf.reduce_max(QT, 1)
        Y = reward + config.gamma * \
            QT_max_action * (1 - terminal)
        Y = tf.stop_gradient(Y)

        loss_batch = clipped_l2(Y, acted_Q)
        loss = tf.reduce_sum(loss_batch, name="loss")

        tf.scalar_summary("losses/loss", loss,
                          collections=[Collection + "_summaries"])
        tf.scalar_summary("losses/loss_0", loss_batch[0],
                          collections=[Collection + "_summaries"])
        tf.scalar_summary("losses/loss_max", tf.reduce_max(loss_batch),
                          collections=[Collection + "_summaries"])
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
            "main/reward_max", tf.reduce_max(reward), collections=[Collection + "_summaries"])

    train_op, grads = graves_rmsprop_optimizer(
        loss, config.learning_rate, 0.95, 0.01, 1)

    for grad, var in grads:
        if grad is True:
            tf.histogram_summary(var.op.name + '/gradients', grad, name=var.op.name +
                                 '/gradients', collections=[Collection + "_summaries"])
    return train_op

def pdqn_train_op(Q, predicted_reward, predicted_next_Q, QT, reward, action, terminal, config, Collection):
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
            clipped_l2(Y, acted_Q), name="Q_loss")

        # note that the target is defined over all actions
        prediction_loss = config.alpha / config.action_num * tf.reduce_sum(clipped_l2(
            predicted_next_Q, QT, grad_clip=config.alpha), name="future_loss")

        predicted_reward_loss = config.alpha * tf.reduce_sum(
            clipped_l2(predicted_reward_action, reward), name="R_loss")

        # maybe add the linear factor 2(DQNR-real_R)(predicted_next_Q-next_Y)
        combined_loss = Q_loss + prediction_loss + predicted_reward_loss

        tf.scalar_summary(
            "losses/Q", Q_loss, collections=[Collection + "_summaries"])
        tf.scalar_summary(
            "losses/predicted", prediction_loss, collections=[Collection + "_summaries"])
        tf.scalar_summary(
            "losses/r", predicted_reward_loss, collections=[Collection + "_summaries"])
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
            "main/max_predicted_Q_0", tf.reduce_max(predicted_next_Q, 1)[0], collections=[Collection + "_summaries"])

        tf.scalar_summary(
            "main/predicted_reward_0", predicted_reward_action[0], collections=[Collection + "_summaries"])
        tf.scalar_summary(
            "main/reward_max", tf.reduce_max(reward), collections=[Collection + "_summaries"])

    train_op, grads = graves_rmsprop_optimizer(
        combined_loss, config.learning_rate, 0.95, 0.01, 1)

    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad, name=var.op.name +
                                 '/gradients', collections=[Collection + "_summaries"])
    return train_op

# -- Primitive ops --
def build_activation_summary(x, Collection, name=None):
    if name:
        tensor_name = name
    else:
        tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x, collections=[Collection + "_summaries"])
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x), collections=[Collection + "_summaries"])


def conv2d(x, W, stride, name):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID", name=name)


def xavier_std(in_size, out_size):
    return np.sqrt(2. / (in_size + out_size))


def get_var(name, size, initializer, Collection):
    w = tf.get_variable(name, size, initializer=initializer,
                        collections=[Collection + "_weights", tf.GraphKeys.VARIABLES])
    if tf.get_variable_scope().reuse == False:
        tf.add_to_collection(Collection + "_summaries",
                             tf.histogram_summary(w.op.name, w))
    return w


def add_conv_layer(head, channels, kernel_size, stride, Collection):
    assert len(head.get_shape()
               ) == 4, "can't add a conv layer to this input"
    layer_name = "conv" + \
        str(len(tf.get_collection(Collection + "_convolutions")))
    tf.add_to_collection(Collection + "_convolutions", layer_name)
    head_channels = head.get_shape().as_list()[3]
    w_size = [kernel_size, kernel_size, head_channels, channels]
    std = xavier_std(head_channels * kernel_size **
                     2, channels * kernel_size**2)

    w = get_var(layer_name + "_W", w_size, initializer=tf.truncated_normal_initializer(
        stddev=std), Collection=Collection)

    new_head = tf.nn.relu(
        conv2d(head, w, stride, name=layer_name), name=layer_name + "_relu")
    build_activation_summary(new_head, Collection)
    return new_head


def add_linear_layer(head, size, Collection, layer_name=None, weight_name=None):
    assert len(head.get_shape()
               ) == 2, "can't add a linear layer to this input"
    if layer_name == None:
        layer_name = "linear" + \
            str(len(tf.get_collection(Collection + "_linears")))
        tf.add_to_collection(Collection + "_linears", layer_name)
    if weight_name == None:
        weight_name = layer_name + "_W"
    head_size = head.get_shape().as_list()[1]
    w_size = [head_size, size]
    std = xavier_std(head_size, size)

    w = get_var(weight_name, w_size, initializer=tf.truncated_normal_initializer(
        stddev=std), Collection=Collection)

    new_head = tf.matmul(head, w, name=layer_name)
    build_activation_summary(new_head, Collection)
    return new_head


def add_relu_layer(head, size, Collection, layer_name=None, weight_name=None):
    if layer_name == None:
        layer_name = "relu" + \
            str(len(tf.get_collection(Collection + "_relus")))
        tf.add_to_collection(Collection + "_relus", layer_name)
    #head = add_linear_layer(
    #    head, size, Collection, layer_name=layer_name+"_linear", weight_name=weight_name)
    head = add_linear_layer(
        head, size, Collection, weight_name=weight_name)
    new_head = tf.nn.relu(head, name=layer_name)
    build_activation_summary(new_head, Collection)
    return new_head


def clipped_l2(y, y_t, grad_clip=1):
    with tf.name_scope("clipped_l2"):
        batch_delta = y - y_t
        batch_delta_abs = tf.abs(batch_delta)
        batch_delta_quadratic = tf.minimum(batch_delta_abs, grad_clip)
        batch_delta_linear = (
            batch_delta_abs - batch_delta_quadratic) * grad_clip
        batch = batch_delta_linear + batch_delta_quadratic**2 / 2
    return batch


def graves_rmsprop_optimizer(loss, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip):
    with tf.name_scope('rmsprop'):
        optimizer = None
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)

        grads = []
        params = []
        for p in grads_and_vars:
            if p[0] == None:
                continue
            grads.append(p[0])
            params.append(p[1])
        #grads = [gv[0] for gv in grads_and_vars]
        #params = [gv[1] for gv in grads_and_vars]
        if gradient_clip > 0:
            grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

        square_grads = [tf.square(grad) for grad in grads]

        avg_grads = [tf.Variable(tf.zeros(var.get_shape()))
                     for var in params]
        avg_square_grads = [tf.Variable(
            tf.zeros(var.get_shape())) for var in params]

        update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + tf.scalar_mul((1 - rmsprop_decay), grad_pair[1]))
                            for grad_pair in zip(avg_grads, grads)]
        update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1])))
                                   for grad_pair in zip(avg_square_grads, grads)]
        avg_grad_updates = update_avg_grads + update_avg_square_grads

        rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
               for avg_grad_pair in zip(avg_grads, avg_square_grads)]

        rms_updates = [grad_rms_pair[0] / grad_rms_pair[1]
                       for grad_rms_pair in zip(grads, rms)]
        train = optimizer.apply_gradients(zip(rms_updates, params))

        return tf.group(train, tf.group(*avg_grad_updates)), grads_and_vars
