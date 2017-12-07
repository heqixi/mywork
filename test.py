import tensorflow as tf 
import numpy as np 
a = tf.placeholder(tf.float32,shape= [1,2])
b = tf.placeholder(tf.float32,shape=[2,1])

y = tf.exp(tf.matmul(a,b))

sess = tf.Session()

print(sess.run(y,feed_dict= {a:(np.asarray([[1],[2]])).T,b:(np.asarray([[1],[2]]))}))
