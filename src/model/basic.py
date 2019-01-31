import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

N = 60
INITIAL_LEARNING_RATE = 1e-2
DECAY_STEPS = 4000
LEARNING_RATE_DECAY_FACTOR = 0.99 # The learning rate decay factor
NUM_EPOCHS = 20000

DATA_DIR = "../../data/"
FREEZE_DIR = '../../freeze/'

def deconv2d(input, channels, stride, is_training, has_act):
    out = tf.layers.conv2d_transpose(input, channels, [3,3], [stride, stride], padding='same', use_bias=not has_act)
    if has_act:
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)
    return out

def model(is_training):
    input = tf.placeholder(tf.float32, [None], name='input')

    out = tf.reshape(input, [-1,1,1,1])
    out = deconv2d(out, 128, 2, is_training, True)
    out = deconv2d(out, 64, 1, is_training, True)
    out = deconv2d(out, 32, 2, is_training, True)
    out = deconv2d(out, 16, 2, is_training, True)
    out = deconv2d(out, 8, 2, is_training, True)
    out = deconv2d(out, 4, 2, is_training, True)
    out = deconv2d(out, 2, 2, is_training, True)
    out = deconv2d(out, 1, 2, is_training, False)
    out = tf.identity(out, name='output')
    return (input, out)

def create_optimizer(learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer

def train():
    input, output = model(True)

    x = []
    y = []
    for i in range(0, 60):
        x.append(i / 120.0)
        y.append(np.expand_dims(plt.imread(DATA_DIR + 'o_' + str(i) + '.png'), axis=2))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                global_step,
                                                DECAY_STEPS,
                                                LEARNING_RATE_DECAY_FACTOR,
                                                staircase=True)
    targets = tf.placeholder(tf.float32,[None,128,128,1])
    tf.losses.add_loss(tf.reduce_mean(tf.square(targets - output)))

    loss = tf.losses.get_total_loss()
    optimizer = create_optimizer(learning_rate).minimize(loss=loss, global_step=global_step)
    init = tf.global_variables_initializer()
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    x_train = np.asarray(x)
    y_train = np.asarray(y)

    logf = open('train.log', 'a')

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)
        save_path = tf.train.latest_checkpoint('./checkpoint')
        if save_path:
            saver.restore(session, save_path)
        for curr_epoch in range(NUM_EPOCHS):
            val_feed = {input: x_train, targets: y_train}
            loss_val, steps, _, _ = session.run([loss, global_step, optimizer, extra_update_ops], val_feed)
            if curr_epoch % 100 == 0:
                saver.save(session, "./checkpoint/basic.model", global_step=steps)
            log = 'Epoch {}/{}, steps = {}, train_cost = {:.6f}'
            print(log.format(curr_epoch + 1, NUM_EPOCHS, steps, loss_val))
            print(log.format(curr_epoch + 1, NUM_EPOCHS, steps, loss_val), file=logf)

def freeze():
    from tensorflow.python.tools import freeze_graph
    with tf.Session() as sess:
        model(False)
        # export graph
        input_graph_name = 'basic.pb'
        tf.train.write_graph(sess.graph, FREEZE_DIR, input_graph_name, as_text=False)

        checkpoint_path = tf.train.latest_checkpoint('./checkpoint')
        # freeze graph
        input_graph_path = FREEZE_DIR + input_graph_name
        input_saver_def_path = ''
        input_binary = True
        output_node_names = 'output'
        restore_op_name = 'save/restore_all'
        filename_tensor_name = 'save/Const:0'
        output_graph_path = FREEZE_DIR + 'basic.pb'
        clear_devices = False
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary, checkpoint_path, output_node_names,
                                  restore_op_name, filename_tensor_name, output_graph_path, clear_devices, '')

def infer():
    with tf.gfile.FastGFile(FREEZE_DIR + "basic.pb", "rb") as f:
        gd = tf.GraphDef()
        gd.ParseFromString(f.read())
        tf.import_graph_def(gd, name='')

    with tf.Session() as sess:
        input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        images = [0.1]
        feed_dict = {input: images}
        targets = sess.run(output, feed_dict=feed_dict)
        im = np.repeat(np.clip(targets[0], 0.0, 1.0), 3, axis=2)
        plt.imsave('infer.png', im)

#train()
#freeze()
infer()