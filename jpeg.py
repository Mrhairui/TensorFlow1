import matplotlib.pyplot as plt
import tensorflow as tf
image = tf.gfile.FastGFile(r'G:\paper\python\kaggle\TensorFlow\123.jpeg', 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image)
    print(img_data.eval())
    plt.imshow(img_data.eval())
    plt.show()
    #
    # encoded_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile(r'path\to\output', 'wb') as f:
    #     f.write(encoded_image.eval())

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    print(img_data.eval())
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    plt.imshow(resized.eval())
    plt.show()



