import numpy as np
import tensorflow as tf 
from load_data import load_data

# generate the fake data 
input_x,labels = load_data("./frequency.csv")
# define the train_batch size 
batch_size = 10

#define the input node and output node 
x = tf.placeholder(tf.float32,shape=(None,4),name = 'x_input')

# define the output 
y_ = tf.placeholder(tf.float32,name='y_output')

# define the paramether 
w = tf.Variable(tf.random_normal([4,1],mean = 0.0 ,stddev= 0.1,seed = 1))
b = tf.Variable(tf.zeros(1))
# difine a node 
temp = tf.matmul(x,w) + b
y = tf.exp(temp)
#define the loss 

loss = tf.reduce_mean(tf.reduce_sum(y-y_*tf.log(y + 0.1)))


#minimize the loss 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# initialize the variables 
init = tf.global_variables_initializer()
# run the module
with tf.Session() as sess:
    sess.run(init)
    for j in range(70):
        for i in range (120):
            _,c  =sess.run([train_step,loss],feed_dict =
                    {x:(np.asarray(input_x[i:i+10])),y_:(np.asarray(labels[i:i+10])).T})
            #print(temp.eval(feed_dict={x:(np.asarray([input_x[i]]))}))
            print("the loss --->",c)
            if ((j % 10) == 0 or j == 69 ):

                print(w.eval())
                print(b.eval())
        #print(y.eval(feed_dict={x:(np.asarray([input_x[i]]))}),"<-->",labels[i])    

