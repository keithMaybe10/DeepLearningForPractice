import tensorflow as tf
import numpy as np
import threading
import time

#线程中运行的程序，这个程序每隔1秒判断是否需要停止并打印线程ID
def MyLoop(coord, worker_id):
    #使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        #随机停止所有的线程
        if np.random.rand() < 0.1:
            print('stop from id: %d\n' % worker_id)
            coord.request_stop()
        else:
            #打印当前线程ID
            print('Working on id: %d\n' % worker_id)
            time.sleep(1)

#声明一个tf.train.Coordinator类来协同多个线程
coord = tf.train.Coordinator()
#创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)]
#启动所有线程
for t in threads:
    t.start()
#等待所有线程退出
coord.join(threads)
