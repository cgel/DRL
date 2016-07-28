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

def createQNetwork(summaryCollection, action_num):
    conv_layer_counter = [0]
    linear_layer_counter = [0]
    weight_list = []

    def add_conv_layer(nn_head, channels, kernel_size, stride):
        assert len(nn_head.get_shape()) == 4, "can't add a conv layer to this input"
        conv_layer_counter[0] += 1
        layer_name = "conv"+str(conv_layer_counter[0])
        nn_head_channels = nn_head.get_shape().as_list()[3]
        w_size = [kernel_size, kernel_size, nn_head_channels, channels]
        w = tf.Variable(tf.truncated_normal(w_size, stddev = xavier_std(nn_head_channels, channels), name=layer_name+"_W_init"), name=layer_name+"_W")
        weight_list.append(w)
        new_head = tf.nn.relu(conv2d(nn_head, w, stride, name=layer_name), name=layer_name+"_relu")
        tf.add_to_collection(summaryCollection, tf.histogram_summary(layer_name+"_relu", new_head))
        return new_head

    def add_linear_layer(nn_head, size):
        assert len(nn_head.get_shape()) == 2, "can't add a linear layer to this input"
        linear_layer_counter[0] +=1
        layer_name = "linear"+str(linear_layer_counter[0])
        nn_head_size = nn_head.get_shape().as_list()[1]
        w = tf.Variable(tf.truncated_normal([nn_head_size, size], stddev = xavier_std(nn_head_size, size), name=layer_name+"_W_init"), name=layer_name+"_W")
        weight_list.append(w)
        new_head = tf.nn.relu(tf.matmul(nn_head, w, name=layer_name), name=layer_name+"_relu")
        tf.add_to_collection(summaryCollection, tf.histogram_summary(layer_name+"_relu", new_head))
        return new_head

    input_state_placeholder = tf.placeholder("float",[None,84,84,4], name=summaryCollection+"/state_placeholder")
    normalized = input_state_placeholder / 256.

    nn_head = add_conv_layer(normalized, channels=32, kernel_size=8, stride=4)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=4, stride=2)
    nn_head = add_conv_layer(nn_head, channels=64, kernel_size=3, stride=1)

    h_conv3_shape = nn_head.get_shape().as_list()
    nn_head = tf.reshape(nn_head, [-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="conv3_flat")

    nn_head = add_linear_layer(nn_head, size=512)
    Q = add_linear_layer(nn_head, size=action_num)

    for weight in weight_list:
        build_activation_summary(weight, summaryCollection)

    return input_state_placeholder, Q, weight_list

def build_train_op(DQN, Y, action, action_num):
    with tf.name_scope("loss"):
        action_one_hot = tf.one_hot(action, action_num, 1., 0., name='action_one_hot')
        DQN_acted = tf.reduce_sum(DQN * action_one_hot, reduction_indices=1, name='DQN_acted')

        #cliping gradient for each example
        delta_grad_clip = 1
        batch_delta = Y - DQN_acted
        batch_delta_abs = tf.abs(batch_delta)
        batch_delta_quadratic = tf.minimum(batch_delta_abs, delta_grad_clip)
        batch_delta_linear = (batch_delta_abs - batch_delta_quadratic)*2
        batch_loss = batch_delta_linear + batch_delta_quadratic**2
        #batch_loss = (Y - DQN_acted)**2
        loss = tf.reduce_mean(batch_loss)
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_average_loss", loss))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_loss_0", batch_loss[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_Y_0", Y[0]))
        tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_actedDQN_0", DQN_acted[0]))
        #tf.add_to_collection("DQN_summaries", tf.scalar_summary("rm_maxDQN_0", tf.reduce_max(DQN[0])))

    opti = tf.train.RMSPropOptimizer(0.00025,0.95,0.95,0.01)
    grads = opti.compute_gradients(loss)

    train_op = opti.apply_gradients(grads) # have to pass global_step ?????

    for grad, var in grads:
        if grad is not None:
            tf.add_to_collection("DQN_summaries", tf.histogram_summary(var.op.name + '/gradients', grad))

    return train_op
