"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. 
Specify the configuration settings at the beginning according to your 
problem.
This script was written for TensorFlow 1.0 and come with a blog post 
you can find here:
  
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert 
contact: f.kratzert(at)gmail.com
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from Vgg16 import Vgg16
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
train_file = './TrainTest/train3.txt'
val_file = './TrainTest/test3.txt'

# Learning params
learning_rate = 0.0001
num_epochs = 50
batch_size = 128

# Network params
dropout_rate = 0.35
num_classes = 2
train_layers = ['fc8', 'fc7']#,'conv5','conv4','conv3']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./finetune_Vgg16T3"
checkpoint_path = "./finetune_Vgg16T3"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
#fc6 = tf.placeholder(tf.float32, [batch_size, 4096],name ='fc6')
#fc7 = tf.placeholder(tf.float32, [batch_size, 4096],name = 'fc7')
#score = tf.variable_pb2(0,tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = Vgg16(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8
fc6 = tf.convert_to_tensor(model.fc6,dtype=tf.float32)
fc7 = tf.convert_to_tensor(model.fc7,dtype=tf.float32)
#fc6 = model.fc6
#prediction = tf.nn.softmax(model.fc8)
#with tf.name_scope("prediction"):
    # create op to calculate softmax
softmax = tf.nn.softmax(score, name="y_pred")
#    y_pred = tf.nn.softmax(score,name="y_pred")
# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
  tf.summary.histogram(var.name, var)
  
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)
  

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, 
                                     horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
# Start Tensorflow session
with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  
  # Add the model graph to TensorBoard
  writer.add_graph(sess.graph)
  
  # Load the pretrained weights into the non-trainable layer
  model.load_original_weights(sess)#
  #model.load_initial_weights(sess)
  
  print("{} Start training...".format(datetime.now()))
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs
  for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step < train_batches_per_epoch:

            #batch_tx, batch_ty = val_generator.next_batch(batch_size)
            #acc = sess.run(y_pred, feed_dict={x: batch_tx,y:batch_ty})



            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys, 
                                          keep_prob: 1-dropout_rate})
            
           # Generate summary with the current batch of data and write to file
            if step%display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                        y: batch_ys, 
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
            step += 1
            
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx, 
                                                y: batch_ty, 
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        #acc = sess.run(y_pred , feed_dict={x: batch_tx,keep_prob: 1})
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        print("{} Saving checkpoint of model...".format(datetime.now()))  
        
        #save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)  
        
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        