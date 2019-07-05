import tensorflow as tf
matrix1 = tf.constant([[2, 2]])
matrix2 = tf.constant([[1],
                       [2]])   # 必须加逗号

product = tf.matmul(matrix1, matrix2)  # np.dot(a, b)
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

with tf.Session() as sess:
    print(sess.run(product))



