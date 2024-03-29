# _*_ coding: utf-8 v  _*_

import  tensorflow as tf
import  tensorflow.examples.tutorials.mnist import input_data

#配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层的节点个数
FC_SIZE = 512

#通过tr.get_variable函数获取变量，在训练神经网络时会创建这些变量
#在 测试时会通过保存的模型加载这些变量的取值
#在变量加载时可以将滑动平均变量重命名，可以直接通过同样的名字在训练时使用变量自身，在测试时使用变量的滑动平均值
#在此函数中也会将变量的正则化损失加入损失集合
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    #当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合
    #使用add_to_collection函数将一个张量加入losses的集合
    if regularizer  != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义卷积神经网络的前向传播过程，参数train用于区分训练过程和测试
#在程序中将用到dropout方法，可以进一步提升模型可靠性并防止过拟合
#dropout只在训练时使用
def inference(input_tensor, train, regularizer):
    #声明第一层卷积层的变量并实现前向传播过程
    #通过使用不同的命名空间来隔离不同层的变量，这可以让每一层中的变量命名只需要考虑在当前层的作用，而不需要担心的重名问题
    #这里定义的卷积层输入为28*28*32的矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biase", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        #使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    #实现第二层池化层的前向传播过程
    #这里选用最大池化层，池化层过滤器的边长为2,
    #使用全0填充且移动步长为2，这一层的输入是上一层的输出，也就是28*28*32，输出为14*14*32的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    #声明第三层卷积层的变量并实现前向传播过程
    #这一层的输入为14*14*32的矩阵，输出为14*14*64的矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        #使用边长为5深度为64的过滤器，过滤器移动的步长为1，且使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    #实现第四层池化层的前向传播过程
    #这一层和第二层的结构是一样的
    #输入为14*14*64的矩阵， 输出为7*7*64的矩阵
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #将第四层池化层的输出转化为第五层全连接层的输入格式
    #第四层的输出为7*7*64的矩阵，第五层全连接层需要的输入格式为向量
    #所以在这里需要将这个7*7*64的矩阵拉成一个变量
    #pool2.get_shape函数可以得到第四层输出矩阵的维度而不需要手工计算
    #因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个bastch中的数据个数
    pool_shape = pool2.get_shape().as_list()

    #计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽和深度的乘积
    #pool_shape[0]是一个batch中数据的个数

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    #通过tf.reshape函数将第四层的输出变为一个batch的向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    #声明第五层全连接的变量并实现前向传播过程
    #这一层的输入是第四层池化层输出矩阵拉直成的一个向量
    #向量长度为一个batch的数据个数3136，输出是一组长度为512的向量
    #引入dropout的概念。dropout在训练时会随机将部分节点的输出改为0,
    #dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好
    #dropout一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    #声明第六层全连接层的变量并实现前向传播过程
    #这一层的输入为一组长度为512的向量，输出为一组长度为10的向量
    #这一层的输出通过Softmax之后就得到了最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        return logit