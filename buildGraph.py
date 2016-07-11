import tensorflow as tf

def build_activation_summary(x, summaryCollection):
    tensor_name = x.op.name
    hs = tf.histogram_summary(tensor_name + '/activations', x)
    ss = tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.add_to_collection(summaryCollection, hs)
    tf.add_to_collection(summaryCollection, ss)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def createQNetwork(summaryCollection, action_num):
    # network weights
    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3,3,64,64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([3136,512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512,action_num])
    b_fc2 = bias_variable([action_num])

    # input layer

    input_state_placeholder = tf.placeholder("float",[None,84,84,4], name=summaryCollection+"/state_placeholder")

    with tf.name_scope("conv1"):
        h_conv1 = tf.nn.relu(conv2d(input_state_placeholder,W_conv1,4) + b_conv1, name="conv1")
        build_activation_summary(h_conv1, summaryCollection)
        #h_pool1 = self.max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2,2) + b_conv2, name="conv2")
        build_activation_summary(h_conv2, summaryCollection)

    with tf.name_scope("conv3"):
        h_conv3 = tf.nn.relu(conv2d(h_conv2,W_conv3,1) + b_conv3, name="conv3")
        build_activation_summary(h_conv3, summaryCollection)

    with tf.name_scope("flatten"):
        h_conv3_shape = h_conv3.get_shape().as_list()
        h_conv3_flat = tf.reshape(h_conv3,[-1, h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]], name="conv3_flat")

    with tf.name_scope("linear1"):
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1, name="linear1")
        build_activation_summary(h_fc1, summaryCollection)

    with tf.name_scope("Qs"):
        Q = tf.add(tf.matmul(h_fc1,W_fc2), b_fc2, name="Qs")
        build_activation_summary(Q, summaryCollection)

    paramList = [W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2]
    for param in paramList:
        build_activation_summary(param, summaryCollection)

    return input_state_placeholder, Q, paramList


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
    return tf.train.RMSPropOptimizer(0.00025,0.95,0.95,0.01).minimize(loss)
