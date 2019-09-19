import tensorflow as tf

example, label = features['i'], features['j']

#一个batch中样例的个数
batch_size = 3

#组合样例的队列过小，出队操作可能会因为没有数据而被阻碍，导致训练效率降低
#组合样例的队列过大，会占用很多的内存资源
capacity = 1000 + 3 * batch_size

#使用tf.train.batch函数组行样例
#[example, label]参数给出了需要组合的元素
#batch_size给出每个batch中样例的个数，capacity给出每个队列的最大容量
example_batch, label_batch = tf.train.batch([example, label], batch_size = batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #获取并打印组合之后的样例
    for i in range(2):
        cur_example_batch, cur_label_batch=sess.run(
            [example_batch, label_batch]
        )
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)