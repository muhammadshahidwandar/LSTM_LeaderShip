""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation using Leader Detectoin Dataset
"""

from __future__ import print_function

import tensorflow as tf
# from tensorflow.re
from tensorflow.contrib import rnn
import os
import scipy.misc
# import pandas as pd
####For Matlab .mat file feature Load####
import scipy.io as sio
import scipy
import numpy as np
import keras
from sklearn.metrics import confusion_matrix
import math

# Training Parameters
train = True
LOGDIR = './summarylayer2_RndStrd256TC3M2_2Layer_Finl300D'
# Num_Epochs = 30
#learnrate = 1e-3
starter_learning_rate = 1e-3
training_steps = 4000  # 116.
batch_size = 20  # 128
display_step = 30  # accuracy and loss on training batch will be displayed on consol after these steps

# Network Parameters
state_size = 256
num_layers = 1
num_input = 4096  # fc6 feature size as a input length (img shape: 4096)
timesteps = 512  # timesteps
num_hidden = 256  # hidden layer num of features
num_classes = 3  # Leader non Leader total classes (0-2 digits(0 no leader, 1 leader, 2 least leader(intermediate class)))
LeaveOneMeeting = [0, 13, 37, 53, 73, 93, 117, 141, 149, 173, 189, 213, 232]
TestVector  = [1,2,3,4,5,6,7,8,9,10,11,12]
Validation  = [5,4,6,7,9,10,12,11,1,3,2,8]
######################################################Training Data Generation########
Label_file = '../labels.mat';  # Labels Matfile name
path = '../Meetingfc6';  # Path Folder for fc6 feature values
Data_size = 0
Y_traine = scipy.io.loadmat(Label_file)  # load label file
fc6filfixd = 'meetingSequence_CNNFeatures_seperatedMatrix'  # common name for all features mat file


#####Load Test and validation Data, Last meeting is concidered as a Test Data
def LoadTestVadlid(strid,Group):
    Y_test = np.array([], dtype='float32')
    X_test = []
    strt   = LeaveOneMeeting[Group - 1]
    if(Group==1):
        strt = LeaveOneMeeting[Group - 1]
        EndPnt = LeaveOneMeeting[Group]-1
    else:
        strt = LeaveOneMeeting[Group - 1]-1
        EndPnt = LeaveOneMeeting[Group]-1
    LengVect = np.array([], dtype='int16')
    for i in range(strt, EndPnt):  # 232):  ### 1 to 232 - Last meeting leave out, for test ranging from 219-232
        fc6ffname = path + '/' + fc6filfixd + str(i + 1) + '.mat'
        fc6Feature = sio.loadmat(fc6ffname);
        tmpr = np.array(fc6Feature['curr_seq_perSample'], dtype=np.float32);
        strtframIdx = 0;
        endframIdx = strtframIdx + timesteps;
        length = len(tmpr);
        Leng  = 0
        while (endframIdx <= length):
            chunk = (tmpr[strtframIdx:endframIdx, 0:4096]).flatten()
            check = Y_traine['labels'][i];
            if (check != 2):
                Y_test = np.append(Y_test, check)
                X_test.append(chunk)

            if (check == 0):
                strtframIdx = strtframIdx + strid;
            else:
                strtframIdx = strtframIdx + int(strid / 2);
            endframIdx = strtframIdx + timesteps;
            Leng = Leng+1
        LengVect = np.append(LengVect, Leng)
    X_test = np.array(X_test, dtype=np.float32)
    return Y_test, X_test,LengVect

####################################
def TrainDatasetLoad(strid, Group1,Group2):
    ###Files to be loaded
    LevStrtG1 = LeaveOneMeeting[Group1 - 1]
    LevEndG1 = LeaveOneMeeting[Group1]
    LevStrtG2 = LeaveOneMeeting[Group2 - 1]
    LevEndG2 = LeaveOneMeeting[Group2]
    #################
    Y_train = np.array([], dtype='float32')  # Empty numpy arrays for Labels
    #Y_test = np.array([], dtype='float32')
    X_train = [];  # Empty structurs for training feature values
    #X_test = [];
    for i in range(0, 230):  # 232):214  ### 1 to 232 - Last meeting leave out, for test ranging from 219-232
        if (((i < LevStrtG1) | (i >= LevEndG1))&((i < LevStrtG2) | (i >= LevEndG2))):
            fc6ffname = path + '/' + fc6filfixd + str(i + 1) + '.mat'
            fc6Feature = sio.loadmat(fc6ffname);
            tmpr = np.array(fc6Feature['curr_seq_perSample'], dtype=np.float32);
            #np.random.shuffle(tmpr)
            strtframIdx = 0;
            endframIdx = strtframIdx + timesteps;
            length = len(tmpr);
            while (endframIdx <= length):
               chunk = (tmpr[strtframIdx:endframIdx, 0:4096]).flatten()
               check = Y_traine['labels'][i];
              # If index is less than 219, make it a part of Training data and label

               if (check != 2):
                    Y_train = np.append(Y_train, check)
                    X_train.append(chunk)
               if (check == 0):
                    strtframIdx = strtframIdx + strid;
               else:
                   strtframIdx = strtframIdx + int(strid / 2);
               endframIdx = strtframIdx + timesteps;
    ####Code for training data shuffling
    ####Code for training data shuffling
    X_train = np.array(X_train, dtype=np.float32)
    Train_Data = np.hstack([X_train, Y_train.reshape([-1, 1])])
    np.random.shuffle(Train_Data)
    #########################################
    Y_train = keras.utils.to_categorical(Train_Data[:, -1], num_classes)
    X_train = Train_Data[:, :-1]
    return Y_train,X_train

# Tensorflow input and output shape
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
dropout = tf.placeholder(tf.float32)
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

tf.summary.histogram("weights", weights['out'])
tf.summary.histogram("biases", biases['out'])

def lstm_rnn_cell(num_units, dropout):
  _cell = tf.nn.rnn_cell.LSTMCell(num_units,state_is_tuple = True,activation=tf.nn.relu)
  _cell = tf.contrib.rnn.DropoutWrapper(_cell, output_keep_prob = dropout)
  return _cell
# tf.summary.histogram("DropOut", weights['out'])

def RNN(x, weights, biases, dropout):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    #lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    #lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    ####################################New Code
    lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(num_hidden, dropout = 1-dropout) for _ in range(num_layers)], state_is_tuple = True)
    lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(num_hidden, dropout = 1-dropout) for _ in range(num_layers)], state_is_tuple = True)
    # Get lstm cell output
    #try:
    #outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
    #                                          dtype=tf.float32)   #tf.nn.bidirectional_dynamic_rnn
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)

    ######
    # lstm_cell2 = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # dropout2  = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, output_keep_prob=0.35)
    # dropout   = tf.nn.rnn_cell.DropoutWrapper
    # Get lstm cell output
    #outputs, states = rnn.static_rnn(dropoutOut, x, dtype=tf.float32)
    #sum = tf.reduce_mean(outputs,axis=0)#.reduce_sum(outputs, axis=1)
    # Linear activation, using rnn inner loop last output
    #act = tf.matmul(sum, weights['out']) + biases['out']
    act = tf.matmul(outputs[-1], weights['out']) + biases['out']
    tf.summary.histogram("activations", act)
    return act


logits = RNN(X, weights, biases, dropout)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

tf.summary.scalar('LOss', loss_op)
######################Modified Part#########################tf.train.AdamOptimizer
global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           4500, 0.90, staircase=False)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9).minimize(loss_op, global_step=global_step)
tf.summary.scalar('Learning Rate', learning_rate)
###########################################################
# Evaluate model (with test logits, for dropout to be disabled)
Y_predict = tf.argmax(prediction, 1)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('Accuracy', accuracy)
# Initialize the variables (i.e. assign their default value)

summ = tf.summary.merge_all()
init = tf.global_variables_initializer()
# Start training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
# sess = tf.Session(config=tf.ConfigProto(
#    allow_soft_placement=True, log_device_placement=True))
# saver = tf.train.Saver()
#Groups = 5 
# global_step = tf.Variable(0, dtype=tf.int32, trainable=False,name='global_step')
with tf.Session() as sess:
    #sess.run([init])
    train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(LOGDIR + '/test')
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    #Y_val,X_val = LoadValidt(180,1)
    #Y_train,X_train = TrainDatasetLoad(256,1)
    for Groups in TestVector:
        #Groups = Groups+1
        train_writer = tf.summary.FileWriter(LOGDIR + '/train'+str(Groups), sess.graph)
        test_writer = tf.summary.FileWriter(LOGDIR + '/test'+str(Groups))
        writer = tf.summary.FileWriter(LOGDIR)
        writer.add_graph(sess.graph)
        sess.run([init])
        Groups2 = Validation[Groups-1]
        print('Currently Serving Group', Groups)
        Y_test,X_test,VecLengT = LoadTestVadlid(400,Groups)
        Y_val,X_val,VecLengV = LoadTestVadlid(400,Groups2)
        Y_train,X_train = TrainDatasetLoad(400,Groups,Groups2)
        strtPont = 0
        lengt = len(X_train)
        testdrop = 0;
        traindrop = 0.30
        Epochs = 0
        MaxGeoM = 0
        GeoMean_val = 0
        #random_array = np.random.randint(0, (lengt - batch_size), training_steps + 1)
        print('length', lengt)

        # for Epochs in range(0,Num_Epochs)
        for step in range(1, training_steps + 1):
            # strtPont  = random_array[step]
            # print('strt point',strtPont)
            batch_x = X_train[strtPont:(strtPont + batch_size)]
            batch_y = Y_train[strtPont:(strtPont + batch_size), :]
            # print('Batch_y is',batch_y)
            # print('shape of batch_x',batch_x.shape)
            # Reshape data to get 256 timestamps and 4096 feature elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            # writer =  tf.summary.FileWriter('./graphs',sess.graph)
            sess.run([optimizer, loss_op], feed_dict={X: batch_x, Y: batch_y, dropout: traindrop})
            if ((strtPont + batch_size) < lengt - batch_size):
                strtPont = strtPont + batch_size;  # For next batch training data access
            else:
                #del X_train, Y_train
                #random_strid = np.random.randint(80, 160, 1)
                #print('step number and stride  is', step, random_strid[0])
                #Y_train, X_train  = TrainDatasetLoad(random_strid[0],Groups,Groups2)
                strtPont = 0;
                lengt = len(X_train)
                Epochs = Epochs + 1
                print('Current Epoch Number', Epochs)

            # print('Batch_y is',batch_y)
            if step % display_step == 0 or step == 1:
                [loss, acc, s] = sess.run([loss_op, accuracy, summ], feed_dict={X: batch_x, Y: batch_y, dropout: testdrop})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Train Accuracy= " + \
                    "{:.3f}".format(acc))
                # Calculate batch loss and accuracy
                # print('Batch_y is',batch_y)
                # [loss, acc] = sess.run([loss_op, accuracy,summ], feed_dict={X: X_test,
                #                                                     Y: keras.utils.to_categorical(Y_test, num_classes),
                #                                                     dropout:testdrop})
                train_writer.add_summary(s, step)
                # print("Step " + str(step) + ", Minibatch Loss= " + \
                #      "{:.4f}".format(loss) + ", Test Accuracy= " + \
                #      "{:.3f}".format(acc))
                # if(acc>0.48):
                length_t = len(X_val)
                X_val2 = X_val.reshape((-1, timesteps, num_input))
                Y_val1 = keras.utils.to_categorical(Y_val, num_classes)
                Y_val2 = Y_val1[0:length_t, :]
                [loss, acc, s] = sess.run([loss_op, accuracy, summ], feed_dict={X: X_val2, Y: Y_val2, dropout: testdrop})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Validation Accuracy= " + \
                    "{:.3f}".format(acc))
                test_writer.add_summary(s, step)
                #print('Now Confusion Matrix')
                if(acc > 30):
                    MaxGeoM = GeoMean_val
                    Predict1 = sess.run(prediction, feed_dict={X: X_val2, dropout: testdrop})
                    np.set_printoptions(threshold=np.nan)
                    prev = 0;
                    Y_pred = np.array([], dtype='int16')
                    Y_labl = np.array([], dtype='int16')
                    for j in VecLengV:
                        Next = j+prev
                        sum1 = Y_val2[prev:Next,:].mean(axis=0)
                        Y_labl = np.append(Y_labl, np.argmax(sum1, 0))
                        sum2 = Predict1[prev:Next,:].mean(axis=0)
                        Y_pred = np.append(Y_pred, np.argmax(sum2, 0))
                        #print('Original and Expected',sum1,sum2)
                        prev = Next
                    confusion_mat = confusion_matrix(Y_labl,Y_pred)
                    print('original and predicted',confusion_mat)
                    C_mat_val = np.matrix(confusion_mat )
                    percntg_val = np.multiply((C_mat_val / C_mat_val.astype(np.float).sum(axis=1)), 100)
                    GeoMean_val = np.power((percntg_val.item(0)) * (percntg_val.item(4)) * (percntg_val.item(8)), 1 / 3)
                    if(GeoMean_val>0):
                        length_t = len(X_test)
                        X_test = X_test.reshape((-1, timesteps, num_input))
                        X_val2 = X_test
                        Y_val1 = keras.utils.to_categorical(Y_test, num_classes)
                        Y_val2 = Y_val1[0:length_t, :]
                        Predict2 = sess.run(prediction, feed_dict={X: X_val2, dropout: testdrop})
                        np.set_printoptions(threshold=np.nan)
                        prev = 0;
                        Y_pred = np.array([], dtype='int16')
                        Y_labl = np.array([], dtype='int16')
                        for j in VecLengT:
                            Next = j+prev
                            #C_mat_val.astype(np.float).sum(axis=1))
                            sum1 = Y_val2[prev:Next,:].mean(axis=0)
                            Y_labl = np.append(Y_labl, np.argmax(sum1, 0))
                            sum2 = Predict1[prev:Next,:].mean(axis=0)
                            Y_pred = np.append(Y_pred, np.argmax(sum2, 0))
                            #print('Original and Expected',sum1,sum2)
                            prev = Next
                        confusion_test = confusion_matrix(Y_labl,Y_pred)
                        C_mat_test = np.matrix(confusion_test)
                        percntg_test = np.multiply((C_mat_test / C_mat_test.astype(np.float).sum(axis=1)), 100)
                        ##########################################################################################
                        strg1 = 'Leave one Meating Validation Group'+ str(Groups)+'\n'
                        file1 = open("MyFile.txt","a")
                        file1.writelines(strg1)
                        for line in C_mat_val:
                            file1.writelines(str(line))
                            file1.writelines('\n')
                        #########################################################################
                        file1.writelines('Leaveone meeting Test Confusion Matrix and Percentage \n')
                        for line in C_mat_test:
                            file1.writelines(str(line))
                            file1.writelines('\n')
                        for line in percntg_test:
                            file1.writelines(str(line))
                            file1.writelines('\n')
                        file1.close()
            #print('Step Completed')
        X_test = X_test.reshape((-1, timesteps, num_input))       
        print("Optimization Finished!")
        print("Testing Accuracy:", \
            sess.run(accuracy,
                    feed_dict={X: X_test, Y: keras.utils.to_categorical(Y_test, num_classes), dropout: testdrop}))
       
        #np.save('confusion_mat12.npy', C_mat)
