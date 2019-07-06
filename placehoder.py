import tensorflow as tf
node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
op = tf.multiply(node1, node2)
with tf.Session() as sess:
    result = sess.run(op, feed_dict = {node1:[3.0], node2:[4.0]})
    print(result)
