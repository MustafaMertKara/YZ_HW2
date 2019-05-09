import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np

DataAdress="YZ_Data/"

Size="50x50"

imgs=np.load(DataAdress+Size+"_data.npy")
labels=np.load(DataAdress+Size+"_labels.npy")

learning_rate=0.01
epoch=500

with tf.name_scope("input"):
    X1=tf.placeholder(tf.float32,(None,len(imgs[0])))
    Y=tf.placeholder(tf.int32,(None))

with tf.name_scope("Dense"):
    hidden1 = tf.layers.dense(X1, 128, tf.nn.leaky_relu)
    hidden2=tf.layers.dense(hidden1, 128, tf.nn.leaky_relu)
    hidden3=tf.layers.dense(hidden2, 128, tf.nn.leaky_relu)

with tf.name_scope("Output"):
    logits=tf.layers.dense(hidden2,3)


with tf.name_scope("loss"):
    xentrophy=tf.losses.sparse_softmax_cross_entropy(Y,logits)
    loss=tf.reduce_mean(xentrophy)

with tf.name_scope("training"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)

with tf.name_scope("eval"):
    correctness=tf.nn.in_top_k(logits,Y,1)
    result=tf.reduce_mean(tf.cast(correctness,tf.float32))*100


init=tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

cv=StratifiedKFold(10,True)
with tf.Session(config=config) as sess:
    Scores = []
    for j in range(10):
        for train,test in cv.split(imgs,labels):
            Train_data=imgs[train]
            Train_label=labels[train]
            Test_data=imgs[test]
            Test_label=labels[test]

            sess.run(init)

            for i in range(epoch):
                sess.run(training_op,feed_dict={X1:Train_data,Y:Train_label})
            Score=sess.run(result,feed_dict={X1:Test_data,Y:Test_label})
            Scores.append(Score)
    print(Scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(Scores), np.std(Scores) * 2))









