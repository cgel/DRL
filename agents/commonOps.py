import tensorflow as tf
import numpy as np

relu_layer_counter = [0]
conv_layer_counter = [0]
linear_layer_counter = [0]
conditional_linear_layer_counter = [0]

# stack_shape must be a list of (out_channels, kernel_size, stride)
def conv_stack(head, stack_shape, config, Collection):
    for layer_shape in stack_shape:
        head = add_conv_layer(head, channels=layer_shape[0],
                          kernel_size=layer_shape[1] , stride=layer_shape[2], Collection=Collection)
    return head

def flatten(head):
    shape = head.get_shape().as_list()
    head = tf.reshape(
        head, [-1, shape[1] * shape[2] * shape[3]])
    return head


def build_scalar_summary(x, Collection, name=None):
    if name:
        tensor_name = name
    else:
        tensor_name = x.op.name
    tf.summary.scalar(tensor_name, tf.squeeze(x), collections=[Collection + "_summaries"])

def build_activation_summary(x, Collection, name=None):
    if name:
        tensor_name = name
    else:
        tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x, collections=[Collection + "_summaries"])
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x), collections=[Collection + "_summaries"])

def build_hist_summary(x, Collection, name=None):
    if name:
        tensor_name = name
    else:
        tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x, collections=[Collection + "_summaries"])


def conv2d(x, W, stride, name):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID", name=name)


def xavier_std(in_size, out_size):
    return np.sqrt(2. / (in_size + out_size))


def get_var(name, size, initializer, Collection):
    w = tf.get_variable(name, size, initializer=initializer,
                        collections=[Collection + "_weights", tf.GraphKeys.GLOBAL_VARIABLES])
    if tf.get_variable_scope().reuse == False:
        tf.add_to_collection(Collection + "_summaries",
                             tf.summary.histogram(w.op.name, w))
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
    build_hist_summary(new_head, Collection)
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
