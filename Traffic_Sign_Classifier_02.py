import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import time
import pickle
import random
import csv

import skimage.exposure
import skimage.io
import tensorflow.contrib.layers
import sklearn.utils

###################################################################
# STEP 0: Load The Data
###################################################################

# Load pickled data
# TODO: Fill this in based on where you saved the training and testing data

training_file = r'traffic-signs-data/train.p'
validation_file=r'traffic-signs-data/valid.p'
testing_file = r'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test,  y_test  = test['features'],  test['labels']

# STEP 1A: Dataset Summary & Exploration
# Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = train['features'].shape[0]

# TODO: Number of testing examples.
n_test  = test['features'].shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = train['features'].shape[1:4]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(train['labels']))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


n_valid = valid['features'].shape[0]
print()
print('**** DATA SUMMARY ****')
print("Number of classes             =", n_classes)
print("Number of training examples   =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples    =", n_test)
print()
print("Image data shape =", image_shape)
print("Image data type  =", type(n_train) , '\n')
print()


# STEP 1B: Dataset Summary & Exploration
# Include an exploratory visualization of the dataset

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.
# %matplotlib inline (commented out, but needed for JUPYTER)

# CSV with sign names
signnames_class=[]
with open('signnames.csv', 'rt') as file_csv:
    read_csv = csv.DictReader(file_csv, delimiter=',')
    for row in read_csv:
        signnames_class.append((row['SignName']))

# Plot 43 classes of images
print('***** TRAINING DATA *****')
fig,ax = plt.subplots()    
for i in range(0,43):
    x_temp = X_train[np.where(y_train==i)]
    number = random.randrange(0,x_temp.shape[0])
    x_temp_rand = x_temp[number]
    ax = fig.add_subplot(5,9,i+1)
    ax.imshow(x_temp_rand)
    ax.set_title(signnames_class[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
plt.show()

#HISTOGRAM of DATA
y_train_hist, y_train_bin_edges = np.histogram(y_train, bins = np.linspace(-0.5,42.5,44))
plt.hist(y_train,bins=np.linspace(-0.5,42.5,44), edgecolor='k', alpha = 0.5, color= 'b')
plt.xlabel('Label ID')
plt.ylabel('Count')
plt.title('Training Data Histogram')
plt.grid(True)
plt.show()

        
# STEP 2A: Design and Test a Model Architecture
#Pre-process the Data Set (normalization, grayscale, etc.)

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# IMAGE PROCESSING
               
imgprocess = True
imgprocess_load_npy = False

if imgprocess:
    X_train2 = X_train[:,:,:,0]*0.299 + X_train[:,:,:,1]*0.587 + X_train[:,:,:,2]*0.114
    X_train2 = X_train2.astype(np.float32)
    X_train2 = X_train2 / 255.0
    for i in range(X_train2.shape[0]):
        print("i=",i)
        X_train2[i] = skimage.exposure.equalize_adapthist( X_train2[i] )
    X_train2 = X_train2 - X_train2.mean()
    X_train2 = np.expand_dims(X_train2, axis=-1)
    X_train = X_train2
    np.save('X_train.npy', X_train)
elif imgprocess_load_npy:
    X_train = np.load('X_train.npy')

if imgprocess:   
    X_valid2 = X_valid[:,:,:,0]*0.299 + X_valid[:,:,:,1]*0.587 + X_valid[:,:,:,2]*0.114
    X_valid2 = X_valid2.astype(np.float32)
    X_valid2 = X_valid2/255.0
    for i in range(X_valid2.shape[0]):
        print("i=",i)
        X_valid2[i] = skimage.exposure.equalize_adapthist( X_valid2[i] )
    X_valid2 = X_valid2 - X_valid2.mean()
    X_valid2 = np.expand_dims(X_valid2, axis=-1) 
    X_valid = X_valid2
    np.save('X_valid.npy', X_valid)
elif imgprocess_load_npy:
    X_valid = np.load('X_valid.npy')
    
if imgprocess:   
    X_test2 = X_test[:,:,:,0]*0.299 + X_test[:,:,:,1]*0.587 + X_test[:,:,:,2]*0.114
    X_test2 = X_test2.astype(np.float32)
    X_test2 = X_test2/255.0
    for i in range(X_test2.shape[0]):
        print("i=",i)
        X_test2[i] = skimage.exposure.equalize_adapthist( X_test2[i] )
    X_test2 = X_test2 - X_test2.mean()
    X_test2 = np.expand_dims(X_test2, axis=-1)
    X_test = X_test2    
    np.save('X_test.npy', X_test)
elif imgprocess_load_npy:
    X_test = np.load('X_test.npy')

# Plot processed images
if imgprocess or imgprocess_load_npy:
    print('***** TRAINING DATA POST PROCESSED *****')
    fig,ax = plt.subplots()    
    for i in range(0,43):
        x_temp = X_train[np.where(y_train==i)]
        number = random.randrange(0,x_temp.shape[0])
        x_temp_rand = x_temp[number]
        ax = fig.add_subplot(5,9,i+1)
        ax.imshow(x_temp_rand.squeeze() , cmap='gray') #squeeze needed to get it to plot
        ax.set_title(signnames_class[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.show()


# DATA AUGMENTATION

data_augment = True

if data_augment:
    print('***** TRAINING DATA AUGMENTATION *****')
    print('X_train size is=',X_train.shape, ' and  y_train size is=',y_train.shape)
    for i in range(43):
        ratio = (y_train_hist.max() - y_train_hist[i]) // y_train_hist[i]
        if ratio > 0:
           print('i=',i,'size=',x_temp.shape,'augmented by=', ratio, ' times')      
           x_temp = X_train[np.where(y_train==i)]
           y_temp = y_train[np.where(y_train==i)]
           for j in range(ratio):
               X_train = np.append(X_train, x_temp,axis=0)
               y_train = np.append(y_train, y_temp,axis=0)
    print('X_train size is=',X_train.shape, ' and  y_train size is=',y_train.shape)
    #HISTOGRAM of augmented DATA
    y_train_hist, y_train_bin_edges = np.histogram(y_train, bins = np.linspace(-0.5,42.5,44))
    plt.hist(y_train,bins=np.linspace(-0.5,42.5,44), edgecolor='k', alpha = 0.5, color= 'b')
    plt.xlabel('Label ID')
    plt.ylabel('Count')
    plt.title('Training Data Histogram (After Augmentation)')
    plt.grid(True)
    plt.show()


# GIVE DATA THE CORRECT NAME FOR TF SESSION

X_train     , y_train      = X_train, y_train
X_validation, y_validation = X_valid, y_valid
X_test      , y_test       = X_test,  y_test

# SHUFFLE THE DATA
# No need to shuffle the data, it's shuffled in TF Session Loop
#X_train, y_train = sklearn.utils.shuffle(X_train, y_train)


# STEP 2B: Design and Test a Model Architecture
# Model Architecture

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1, name='act1')

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2, name='act2')

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = tensorflow.contrib.layers.flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma)) #ANDRES
    fc3_b  = tf.Variable(tf.zeros(43)) #ANDRES
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
    
  
# STEP 2C: Design and Test a Model Architecture  
# Train, Validate and Test the Model
  
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# MODEL PARAMS
EPOCHS = 30
BATCH_SIZE = 128

# FEATURES and LABELS
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)  # 43 TRAFFIC SIGN CLASSES

# TRAINING a PIPELINE
rate = 0.001
logits = LeNet(x)
cross_entropy      = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation     = tf.reduce_mean(cross_entropy)
optimizer          = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# MODEL EVALUATION
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# TRAIN THE MODEL
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

t_start=time.time()

accuracy_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        accuracy_list.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

t_end = time.time()  
print("Training Time = ", t_end - t_start)

# Plot for Accuracy vs Epochs
plt.plot(np.arange(EPOCHS)+1, accuracy_list)
plt.ylim((0.0 , 1.0))
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY')
plt.show()

# MODEL ACCURACY USING TEST SET
loader = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loader.restore(sess, './lenet')
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    
###################################################################
# Step 3: Test a Model on New Images    
###################################################################

#1-Load and Output the Images
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

image1 = skimage.io.imread("traffic-signs-data/image1.jpg")
image2 = skimage.io.imread("traffic-signs-data/image2.jpg")
image3 = skimage.io.imread("traffic-signs-data/image3.jpg")
image4 = skimage.io.imread("traffic-signs-data/image4.jpg")
image5 = skimage.io.imread("traffic-signs-data/image5.jpg")

X_images = np.array([image1, image2, image3, image4, image5])

print('FIVE IMAGES FOR TESTING')
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_images[i]) 
    plt.title(X_images[i])
    plt.axis('off')
plt.show()

#2-Predict the Sign Type for Each Image
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

# Image processing - Grayscale, contrast, center over mean
X_images2 = X_images[:,:,:,0]*0.299 + X_images[:,:,:,1]*0.587 + X_images[:,:,:,2]*0.114
X_images2 = X_images2.astype(np.float32)
X_images2 = X_images2 / 255.0
for i in range(X_images2.shape[0]):
    X_images2[i] = skimage.exposure.equalize_adapthist( X_images2[i] )
X_images2 = X_images2 - X_images2.mean()
X_images2 = np.expand_dims(X_images2, axis=-1)

# This function is to predict the class of a given image
def test_net(X_data, sess):
    pred_sign = sess.run(tf.argmax(logits, 1), feed_dict={x: X_data})
    return pred_sign
loader = tf.train.Saver()    
with tf.Session() as sess:
    loader.restore(sess, './lenet')
    X_images_class=test_net(X_images2, sess)

# Plot the 5 images with their correspoinding prediction
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_images[i]) 
    plt.title(signnames_class[X_images_class[i]])
    plt.axis('off')
plt.show()

#3-Analyze Performance
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

# This function is to output the top5 prediction probabilities for an image

logits_placeholder = tf.placeholder(tf.float32)
softmax = tf.nn.softmax(logits)
def prob5_net(X_data, sess):
    softmaxProb = sess.run(softmax, feed_dict={x: X_data})
    prob5 = sess.run(tf.nn.top_k(tf.constant(softmaxProb), k=5))
    return prob5

loader = tf.train.Saver()    
with tf.Session() as sess:
    loader.restore(sess, './lenet')
    X_images_prob5=prob5_net(X_images2, sess)

print('****** TOP 5 PROBABILITY SOFTMAX ******')    
print(X_images_prob5[0])  
print('****** TOP 5 PROBABILITY CLASS   ******') 
print(X_images_prob5[1])   
print()

print('****** TOP 5 PROBABILITY PERCENT *****')
print()
prob5 = X_images_prob5[0]
indi5 = X_images_prob5[1]
for i in range(X_images.shape[0]):
    plt.imshow(X_images[i])
    plt.show()
    prob = prob5[i]
    ind  = indi5[i]
    prob = np.array( [ num for num in prob if num >= 0 ] )
    prob = prob/prob.sum()*100
    for j in range(len(prob)):
        print(prob[j],'% class=',ind[j],' ',signnames_class[ind[j]])
    print()
  
    
# 4
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
    plt.show()
     
mu = 0
sigma = 0.1
# SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
conv1_b = tf.Variable(tf.zeros(6))
conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
# SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
conv2_b = tf.Variable(tf.zeros(16))
conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

print()
print('***** FEATURE MAP - CONV1 *****')
with tf.Session() as sess:
    saver.restore(sess, './lenet')
    sess.run(tf.global_variables_initializer())
    tf_conv1 = sess.run(conv1, feed_dict={x: X_train})
    outputFeatureMap(X_images2, conv1)  
print()    
print('***** FEATURE MAP - CONV2 *****')
with tf.Session() as sess:
    saver.restore(sess, './lenet')
    sess.run(tf.global_variables_initializer())
    tf_conv2 = sess.run(conv2, feed_dict={x: X_train})
    outputFeatureMap(X_images2, conv2)  
