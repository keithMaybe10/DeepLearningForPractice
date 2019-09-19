import tensorflow as tf

#声明一个元素个数为100的FIFO的队列，类型为实数
queue = tf.FIFOQueue(100, "float")

#定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

#使用tf.train.QueueRunner来创建多个线程运行队列的入列操作
#tf.train.QueueRunner的第一个参数给出了被操作的队列
# [enqueue_op] * 5表示需要启动五个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

#将定义过的QueueRunner加入Tensorflow计算图上指定的集合
#因未指定集合，故加入到默认集合tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)

#定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    #使用tf.train.Coordinator来协同启动的线程
    coord = tf.train.Coordinator()
    #使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runners来启动所有线程
    threads = tf.train.start_queue_runners(sess=sess, coord = coord)
    #获取队列中的取值
    for _ in range(3):
        print(sess.run(out_tensor)[0])

    #使用tf.train.Coordinator来停止所有的线程
    coord.request_stop()
    coord.join(threads)