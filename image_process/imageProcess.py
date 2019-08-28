import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#随机调整给定图像的色彩、对比度、饱和度和色相的顺序
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    
    return tf.clip_by_value(image, 0.0, 1.0)

#给定一张解码后的图像，目标图像的尺寸以及图像上的标注框
#函数可以对给出的图像进行预处理
#函数输入图像是图像识别问题中原始的训练图像，而输出则是神经网络模型的输入层
#这里只出来了模型的训练数据，对于预测的数据，一般不需要使用随机变换的步骤
def preprocess_for_train(image, height, width, bbox):
    