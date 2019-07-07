from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
MOVING_AVERAGE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):

    if avg_class is None:
        # is 而不是==，这里是为什么呢？is是id相同，==是值相同，==是eq函数来的，所以可以被重载
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # tf.nn.relu 激活函数，前向传播中的一步
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)+ avg_class.average(biases2))


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
    # 定义x,y，一般使用placeholder函数，后面可以自定义batch，
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    # stddev 是标准差， 也可以指定mean默认是0，random_normal ，truncated_normal (随机取值不同)
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # Variable 两种定义形式,针对w是利用随机变量，针对b是常数

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 前面是定义参数,接下来就是前向传播
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 下面式滑动平均的定义，首先通过一次来计算影子变量的值
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_average_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算loss，对于多分类问题使用的是交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 这里要注意sparse_softmax_cross_entropy_with_logits，函数以及softmax_cross_entropy_with_logits前面的函数要实现onehot以及softmax，但是后面的label已经onehot了
    cross_entropy_mean = tf.reduce_mean(cross_entropy)


    # 计算正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    # 计算总的损失函数
    loss = cross_entropy_mean + regularization
    #计算学习速率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    #优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32 ))


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images, y_:mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict = validate_feed)
                print('after %d training steps, validation accuracy''using average model is %g' % (i, validate_acc))
                total_cross = sess.run(cross_entropy_mean, validate_feed)
                print(total_cross)
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict = {x:xs, y_:ys}) # c
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('after %d training steps' 'model is %g'  %(TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets(r'G:\paper\python\kaggle\TensorFlow\MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()




















