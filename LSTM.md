
# <a title="Activity Recognition" href="https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition" > LSTMs for Human Activity Recognition</a>

Human Activity Recognition (HAR) using smartphones dataset and an LSTM RNN. Classifying the type of movement amongst six categories:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.

Compared to a classical approach, using a Recurrent Neural Networks (RNN) with Long Short-Term Memory cells (LSTMs) require no or almost no feature engineering. Data can be fed directly into the neural network who acts like a black box, modeling the problem correctly. [Other research](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names) on the activity recognition dataset can use a big amount of feature engineering, which is rather a signal processing approach combined with classical data science techniques. The approach here is rather very simple in terms of how much was the data preprocessed. 

Let's use Google's neat Deep Learning library, TensorFlow, demonstrating the usage of an LSTM, a type of Artificial Neural Network that can process sequential data / time series. 

## Video dataset overview

Follow this link to see a video of the 6 activities recorded in the experiment with one of the participants:

<p align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=XOEN9W05_4A
" target="_blank"><img src="http://img.youtube.com/vi/XOEN9W05_4A/0.jpg" 
alt="Video of the experiment" width="400" height="300" border="10" /></a>
  <a href="https://youtu.be/XOEN9W05_4A"><center>[Watch video]</center></a>
</p>

## Details about the input data

I will be using an LSTM on the data to learn (as a cellphone attached on the waist) to recognise the type of activity that the user is doing. The dataset's description goes like this:

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. 

That said, I will use the almost raw data: only the gravity effect has been filtered out of the accelerometer  as a preprocessing step for another 3D feature as an input to help learning. If you'd ever want to extract the gravity by yourself, you could fork my code on using a [Butterworth Low-Pass Filter (LPF) in Python](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) and edit it to have the right cutoff frequency of 0.3 Hz which is a good frequency for activity recognition from body sensors.

## What is an RNN?

As explained in [this article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), an RNN takes many input vectors to process them and output other vectors. It can be roughly pictured like in the image below, imagining each rectangle has a vectorial depth and other special hidden quirks in the image below. **In our case, the "many to one" architecture is used**: we accept time series of [feature vectors](https://www.quora.com/What-do-samples-features-time-steps-mean-in-LSTM/answer/Guillaume-Chevalier-2) (one vector per [time step](https://www.quora.com/What-do-samples-features-time-steps-mean-in-LSTM/answer/Guillaume-Chevalier-2)) to convert them to a probability vector at the output for classification. Note that a "one to one" architecture would be a standard feedforward neural network. 

> <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/" ><img src="http://karpathy.github.io/assets/rnn/diags.jpeg" /></a>
> http://karpathy.github.io/2015/05/21/rnn-effectiveness/

## What is an LSTM?

An LSTM is an improved RNN. It is more complex, but easier to train, avoiding what is called the vanishing gradient problem. I recommend [this article](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for you to learn more on LSTMs.


## Results 

Scroll on! Nice visuals awaits. 


```python
# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

import os
```


```python
# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 

```

## Let's start by downloading the data: 


```python
## Note: Linux bash commands start with a "!" inside those "ipython notebook" cells
#
DATA_PATH = "data/"
#
#!pwd && ls
#os.chdir(DATA_PATH)
#!pwd && ls
#
#!python download_dataset.py
#
#!pwd && ls
#os.chdir("..")
#!pwd && ls
#
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)
#
```

    
    Dataset is now located at: data/UCI HAR Dataset/
    

## Preparing dataset:


```python
TRAIN = "train/"
TEST = "test/"


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    
    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

```

## Additionnal Parameters:

Here are some core parameter definitions for the training. 

For example, the whole neural network's structure could be summarised by enumerating those parameters and the fact that two LSTM are used one on top of another (stacked) output-to-input as hidden layers through time steps. 


```python
# Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure

n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)


# Training 

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

```

    Some useful info to get an insight on dataset's shape and normalisation:
    (X shape, y shape, every X's mean, every X's standard deviation)
    (2947, 128, 9) (2947, 1) 0.09913992 0.39567086
    The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.
    


```python
print(('X_train: {}').format(X_train.shape))
print(('X_test: {}').format(X_test.shape))
print(('y_train: {}').format(y_train.shape))
print(('y_test: {}').format(y_test.shape))


##128: readings/window
##[acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "total_acc_x", "total_acc_y" , "total_acc_z"]
##
##["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"] 
```

    X_train: (7352, 128, 9)
    X_test: (2947, 128, 9)
    y_train: (7352, 1)
    y_test: (2947, 1)
    


```python
X_train[0][0]
```




    array([ 1.808515e-04,  1.076681e-02,  5.556068e-02,  3.019122e-02,
            6.601362e-02,  2.285864e-02,  1.012817e+00, -1.232167e-01,
            1.029341e-01], dtype=float32)




```python
X_train[0][1]
```




    array([ 0.01013856,  0.00657948,  0.05512483,  0.04371071,  0.04269897,
            0.01031572,  1.022833  , -0.1268756 ,  0.1056872 ], dtype=float32)




```python
X_train[0][0]
```




    array([ 1.808515e-04,  1.076681e-02,  5.556068e-02,  3.019122e-02,
            6.601362e-02,  2.285864e-02,  1.012817e+00, -1.232167e-01,
            1.029341e-01], dtype=float32)



## Utility functions for training:


```python
def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier, 
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes 
    # e.g.: 
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

```

## Let's get serious and build the neural network:


```python
################
# n_steps: 128   readings / window
# n_input: 9    [acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "total_acc_x", "total_acc_y" , "total_acc_z"]

# n_classes: 6  ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"] 
# n_hidden: 32

#training_data_count: 7352
#test_data_count: 2947

#learning_rate: 0.0025
#lambda_loss_amount: 0.0015
# training_iters: 2205600

#batch_size: 1500
#display_iter: 30000

################


# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])  
y = tf.placeholder(tf.float32, [None, n_classes])         

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
```

    WARNING:tensorflow:From c:\users\guest\appdata\local\programs\python\python37\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    


```python
weights
```


```python
# prediction
pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation:
#################################

# L2 loss prevents this overkill neural network to overfit the data
l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()) 

# Softmax loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

    
    WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From <ipython-input-10-86e187028df0>:22: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From <ipython-input-10-86e187028df0>:24: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.
    WARNING:tensorflow:From <ipython-input-10-86e187028df0>:26: static_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `keras.layers.RNN(cell, unroll=True)`, which is equivalent to this API
    WARNING:tensorflow:From <ipython-input-12-1bc9e0da0660>:7: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    

## Hooray, now train the neural network:


```python
# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

```

    Training iter #1500:   Batch Loss = 2.953793, Accuracy = 0.17533333599567413
    PERFORMANCE ON TEST SET: Batch Loss = 2.6231753826141357, Accuracy = 0.2775703966617584
    Training iter #30000:   Batch Loss = 1.390238, Accuracy = 0.6726666688919067
    PERFORMANCE ON TEST SET: Batch Loss = 1.4284675121307373, Accuracy = 0.6399728655815125
    Training iter #60000:   Batch Loss = 1.148292, Accuracy = 0.7846666574478149
    PERFORMANCE ON TEST SET: Batch Loss = 1.2291245460510254, Accuracy = 0.7434679269790649
    Training iter #90000:   Batch Loss = 0.940727, Accuracy = 0.874666690826416
    PERFORMANCE ON TEST SET: Batch Loss = 1.2538843154907227, Accuracy = 0.773668110370636
    Training iter #120000:   Batch Loss = 0.822707, Accuracy = 0.9006666541099548
    PERFORMANCE ON TEST SET: Batch Loss = 1.0351510047912598, Accuracy = 0.8564642071723938
    Training iter #150000:   Batch Loss = 0.712458, Accuracy = 0.9326666593551636
    PERFORMANCE ON TEST SET: Batch Loss = 0.9602556228637695, Accuracy = 0.8873430490493774
    Training iter #180000:   Batch Loss = 0.795427, Accuracy = 0.887333333492279
    PERFORMANCE ON TEST SET: Batch Loss = 0.9436272978782654, Accuracy = 0.8747879266738892
    Training iter #210000:   Batch Loss = 0.742593, Accuracy = 0.9133333563804626
    PERFORMANCE ON TEST SET: Batch Loss = 0.8773400783538818, Accuracy = 0.8978622555732727
    Training iter #240000:   Batch Loss = 0.628422, Accuracy = 0.9546666741371155
    PERFORMANCE ON TEST SET: Batch Loss = 0.9239457845687866, Accuracy = 0.8897183537483215
    Training iter #270000:   Batch Loss = 0.631237, Accuracy = 0.9513333439826965
    PERFORMANCE ON TEST SET: Batch Loss = 1.0420701503753662, Accuracy = 0.8306752443313599
    Training iter #300000:   Batch Loss = 0.602117, Accuracy = 0.9706666469573975
    PERFORMANCE ON TEST SET: Batch Loss = 0.8353958129882812, Accuracy = 0.900237500667572
    Training iter #330000:   Batch Loss = 0.617571, Accuracy = 0.972000002861023
    PERFORMANCE ON TEST SET: Batch Loss = 0.8414380550384521, Accuracy = 0.9012554883956909
    Training iter #360000:   Batch Loss = 0.615766, Accuracy = 0.9539999961853027
    PERFORMANCE ON TEST SET: Batch Loss = 0.785632848739624, Accuracy = 0.9039701223373413
    Training iter #390000:   Batch Loss = 0.619717, Accuracy = 0.9433333277702332
    PERFORMANCE ON TEST SET: Batch Loss = 0.790411651134491, Accuracy = 0.9043094515800476
    Training iter #420000:   Batch Loss = 0.539221, Accuracy = 0.9646666646003723
    PERFORMANCE ON TEST SET: Batch Loss = 0.7727307081222534, Accuracy = 0.9046487808227539
    Training iter #450000:   Batch Loss = 0.551677, Accuracy = 0.9473333358764648
    PERFORMANCE ON TEST SET: Batch Loss = 0.7606834173202515, Accuracy = 0.9019341468811035
    Training iter #480000:   Batch Loss = 0.556617, Accuracy = 0.9393333196640015
    PERFORMANCE ON TEST SET: Batch Loss = 0.7737100720405579, Accuracy = 0.8998982310295105
    Training iter #510000:   Batch Loss = 0.605913, Accuracy = 0.9446666836738586
    PERFORMANCE ON TEST SET: Batch Loss = 0.772608757019043, Accuracy = 0.8808958530426025
    Training iter #540000:   Batch Loss = 0.839322, Accuracy = 0.8199999928474426
    PERFORMANCE ON TEST SET: Batch Loss = 0.9421875476837158, Accuracy = 0.7964031100273132
    Training iter #570000:   Batch Loss = 0.598832, Accuracy = 0.9153333306312561
    PERFORMANCE ON TEST SET: Batch Loss = 0.6901009678840637, Accuracy = 0.9012554883956909
    Training iter #600000:   Batch Loss = 0.568524, Accuracy = 0.9173333048820496
    PERFORMANCE ON TEST SET: Batch Loss = 0.6650387048721313, Accuracy = 0.9039701223373413
    Training iter #630000:   Batch Loss = 0.474736, Accuracy = 0.9886666536331177
    PERFORMANCE ON TEST SET: Batch Loss = 0.679368257522583, Accuracy = 0.9029521346092224
    Training iter #660000:   Batch Loss = 0.467792, Accuracy = 0.9753333330154419
    PERFORMANCE ON TEST SET: Batch Loss = 0.6994653940200806, Accuracy = 0.9110960364341736
    Training iter #690000:   Batch Loss = 0.444873, Accuracy = 0.9853333234786987
    PERFORMANCE ON TEST SET: Batch Loss = 0.66352379322052, Accuracy = 0.8978622555732727
    Training iter #720000:   Batch Loss = 0.501981, Accuracy = 0.9599999785423279
    PERFORMANCE ON TEST SET: Batch Loss = 0.6631295084953308, Accuracy = 0.9032914638519287
    Training iter #750000:   Batch Loss = 0.512402, Accuracy = 0.949999988079071
    PERFORMANCE ON TEST SET: Batch Loss = 0.6436066627502441, Accuracy = 0.9100780487060547
    Training iter #780000:   Batch Loss = 0.421465, Accuracy = 0.9746666550636292
    PERFORMANCE ON TEST SET: Batch Loss = 0.6314971446990967, Accuracy = 0.910417377948761
    Training iter #810000:   Batch Loss = 0.431740, Accuracy = 0.9520000219345093
    PERFORMANCE ON TEST SET: Batch Loss = 0.6311908960342407, Accuracy = 0.9121140241622925
    Training iter #840000:   Batch Loss = 0.497780, Accuracy = 0.9399999976158142
    PERFORMANCE ON TEST SET: Batch Loss = 0.7156990766525269, Accuracy = 0.8825924396514893
    Training iter #870000:   Batch Loss = 0.434605, Accuracy = 0.9513333439826965
    PERFORMANCE ON TEST SET: Batch Loss = 0.6532403230667114, Accuracy = 0.8998982310295105
    Training iter #900000:   Batch Loss = 0.398939, Accuracy = 0.9786666631698608
    PERFORMANCE ON TEST SET: Batch Loss = 0.6345514059066772, Accuracy = 0.9005768299102783
    Training iter #930000:   Batch Loss = 0.462192, Accuracy = 0.9266666769981384
    PERFORMANCE ON TEST SET: Batch Loss = 0.6197799444198608, Accuracy = 0.9066847562789917
    Training iter #960000:   Batch Loss = 0.447610, Accuracy = 0.9319999814033508
    PERFORMANCE ON TEST SET: Batch Loss = 0.6140636205673218, Accuracy = 0.9107567071914673
    Training iter #990000:   Batch Loss = 0.376879, Accuracy = 0.9746666550636292
    PERFORMANCE ON TEST SET: Batch Loss = 0.6003590822219849, Accuracy = 0.9131320118904114
    Training iter #1020000:   Batch Loss = 0.423405, Accuracy = 0.968666672706604
    PERFORMANCE ON TEST SET: Batch Loss = 0.6026033163070679, Accuracy = 0.9032914638519287
    Training iter #1050000:   Batch Loss = 0.608832, Accuracy = 0.9259999990463257
    PERFORMANCE ON TEST SET: Batch Loss = 0.7934855222702026, Accuracy = 0.8320325613021851
    Training iter #1080000:   Batch Loss = 0.446585, Accuracy = 0.9620000123977661
    PERFORMANCE ON TEST SET: Batch Loss = 0.5998634696006775, Accuracy = 0.8866643905639648
    Training iter #1110000:   Batch Loss = 0.444704, Accuracy = 0.9453333616256714
    PERFORMANCE ON TEST SET: Batch Loss = 0.5815087556838989, Accuracy = 0.9026128053665161
    Training iter #1140000:   Batch Loss = 0.415241, Accuracy = 0.9493333101272583
    PERFORMANCE ON TEST SET: Batch Loss = 0.5818965435028076, Accuracy = 0.9077027440071106
    Training iter #1170000:   Batch Loss = 0.362479, Accuracy = 0.9633333086967468
    PERFORMANCE ON TEST SET: Batch Loss = 0.5826781988143921, Accuracy = 0.907024085521698
    Training iter #1200000:   Batch Loss = 0.375136, Accuracy = 0.9513333439826965
    PERFORMANCE ON TEST SET: Batch Loss = 0.5515061616897583, Accuracy = 0.9202578663825989
    Training iter #1230000:   Batch Loss = 0.361469, Accuracy = 0.9520000219345093
    PERFORMANCE ON TEST SET: Batch Loss = 0.5665402412414551, Accuracy = 0.9043094515800476
    Training iter #1260000:   Batch Loss = 0.336754, Accuracy = 0.972000002861023
    PERFORMANCE ON TEST SET: Batch Loss = 0.5660369396209717, Accuracy = 0.9093993902206421
    Training iter #1290000:   Batch Loss = 0.407423, Accuracy = 0.9240000247955322
    PERFORMANCE ON TEST SET: Batch Loss = 0.6527124643325806, Accuracy = 0.8802171945571899
    Training iter #1320000:   Batch Loss = 0.395506, Accuracy = 0.9333333373069763
    PERFORMANCE ON TEST SET: Batch Loss = 0.5408645868301392, Accuracy = 0.907024085521698
    Training iter #1350000:   Batch Loss = 0.348532, Accuracy = 0.9520000219345093
    PERFORMANCE ON TEST SET: Batch Loss = 0.5329444408416748, Accuracy = 0.9083814024925232
    Training iter #1380000:   Batch Loss = 0.307715, Accuracy = 0.9739999771118164
    PERFORMANCE ON TEST SET: Batch Loss = 0.5918770432472229, Accuracy = 0.9005768299102783
    Training iter #1410000:   Batch Loss = 0.294988, Accuracy = 0.9766666889190674
    PERFORMANCE ON TEST SET: Batch Loss = 0.5271384716033936, Accuracy = 0.9087207317352295
    Training iter #1440000:   Batch Loss = 0.311644, Accuracy = 0.9700000286102295
    PERFORMANCE ON TEST SET: Batch Loss = 0.5426385998725891, Accuracy = 0.9093993902206421
    Training iter #1470000:   Batch Loss = 0.345800, Accuracy = 0.9539999961853027
    PERFORMANCE ON TEST SET: Batch Loss = 0.5396025776863098, Accuracy = 0.9175432920455933
    Training iter #1500000:   Batch Loss = 0.365895, Accuracy = 0.9440000057220459
    PERFORMANCE ON TEST SET: Batch Loss = 0.49452540278434753, Accuracy = 0.9182218909263611
    Training iter #1530000:   Batch Loss = 0.317471, Accuracy = 0.9626666903495789
    PERFORMANCE ON TEST SET: Batch Loss = 0.5193567872047424, Accuracy = 0.9012554883956909
    Training iter #1560000:   Batch Loss = 0.310455, Accuracy = 0.9520000219345093
    PERFORMANCE ON TEST SET: Batch Loss = 0.530788242816925, Accuracy = 0.9053274393081665
    Training iter #1590000:   Batch Loss = 0.347184, Accuracy = 0.9353333115577698
    PERFORMANCE ON TEST SET: Batch Loss = 0.5066947937011719, Accuracy = 0.9060060977935791
    Training iter #1620000:   Batch Loss = 0.302575, Accuracy = 0.9626666903495789
    PERFORMANCE ON TEST SET: Batch Loss = 0.46478787064552307, Accuracy = 0.9202578663825989
    Training iter #1650000:   Batch Loss = 0.334989, Accuracy = 0.9306666851043701
    PERFORMANCE ON TEST SET: Batch Loss = 0.49957743287086487, Accuracy = 0.910417377948761
    Training iter #1680000:   Batch Loss = 0.339877, Accuracy = 0.9293333292007446
    PERFORMANCE ON TEST SET: Batch Loss = 0.4995911717414856, Accuracy = 0.9097387194633484
    Training iter #1710000:   Batch Loss = 0.324191, Accuracy = 0.9340000152587891
    PERFORMANCE ON TEST SET: Batch Loss = 0.506732702255249, Accuracy = 0.9127926826477051
    Training iter #1740000:   Batch Loss = 0.266342, Accuracy = 0.9700000286102295
    PERFORMANCE ON TEST SET: Batch Loss = 0.4969051778316498, Accuracy = 0.9117746949195862
    Training iter #1770000:   Batch Loss = 0.313694, Accuracy = 0.9359999895095825
    PERFORMANCE ON TEST SET: Batch Loss = 0.6065923571586609, Accuracy = 0.8778418898582458
    Training iter #1800000:   Batch Loss = 0.297305, Accuracy = 0.9633333086967468
    PERFORMANCE ON TEST SET: Batch Loss = 0.47391873598098755, Accuracy = 0.910417377948761
    Training iter #1830000:   Batch Loss = 0.305861, Accuracy = 0.9566666483879089
    PERFORMANCE ON TEST SET: Batch Loss = 0.45829862356185913, Accuracy = 0.9267051219940186
    Training iter #1860000:   Batch Loss = 0.318779, Accuracy = 0.940666675567627
    PERFORMANCE ON TEST SET: Batch Loss = 0.4615838825702667, Accuracy = 0.9216151833534241
    Training iter #1890000:   Batch Loss = 0.259661, Accuracy = 0.9646666646003723
    PERFORMANCE ON TEST SET: Batch Loss = 0.4754868745803833, Accuracy = 0.917203962802887
    Training iter #1920000:   Batch Loss = 0.257494, Accuracy = 0.9666666388511658
    PERFORMANCE ON TEST SET: Batch Loss = 0.48739445209503174, Accuracy = 0.8924329876899719
    Training iter #1950000:   Batch Loss = 0.275151, Accuracy = 0.9546666741371155
    PERFORMANCE ON TEST SET: Batch Loss = 0.4477614760398865, Accuracy = 0.9175432920455933
    Training iter #1980000:   Batch Loss = 0.284960, Accuracy = 0.9366666674613953
    PERFORMANCE ON TEST SET: Batch Loss = 0.44053006172180176, Accuracy = 0.9148286581039429
    Training iter #2010000:   Batch Loss = 0.246638, Accuracy = 0.9633333086967468
    PERFORMANCE ON TEST SET: Batch Loss = 0.453444242477417, Accuracy = 0.913810670375824
    Training iter #2040000:   Batch Loss = 0.356012, Accuracy = 0.8913333415985107
    PERFORMANCE ON TEST SET: Batch Loss = 0.4567292332649231, Accuracy = 0.9212758541107178
    Training iter #2070000:   Batch Loss = 0.288302, Accuracy = 0.9340000152587891
    PERFORMANCE ON TEST SET: Batch Loss = 0.47529834508895874, Accuracy = 0.9066847562789917
    Training iter #2100000:   Batch Loss = 0.214549, Accuracy = 0.9779999852180481
    PERFORMANCE ON TEST SET: Batch Loss = 0.467652291059494, Accuracy = 0.910417377948761
    Training iter #2130000:   Batch Loss = 0.265729, Accuracy = 0.9486666917800903
    PERFORMANCE ON TEST SET: Batch Loss = 0.47667863965034485, Accuracy = 0.8863250613212585
    Training iter #2160000:   Batch Loss = 0.242053, Accuracy = 0.9760000109672546
    PERFORMANCE ON TEST SET: Batch Loss = 0.4038800299167633, Accuracy = 0.9212758541107178
    Training iter #2190000:   Batch Loss = 0.264006, Accuracy = 0.9760000109672546
    PERFORMANCE ON TEST SET: Batch Loss = 0.5201723575592041, Accuracy = 0.8771632313728333
    Optimization Finished!
    

## Training is good, but having visual insight is even better:

Okay, let's plot this simply in the notebook for now.


```python
# (Inline plots: )
%matplotlib inline

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()
```

## And finally, the multi-class confusion matrix and metrics!


```python
# Results

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
```


```python
sess.close()
```

## Conclusion

Outstandingly, **the final accuracy is of 91%**! And it can peak to values such as 93.25%, at some moments of luck during the training, depending on how the neural network's weights got initialized at the start of the training, randomly. 

This means that the neural networks is almost always able to correctly identify the movement type! Remember, the phone is attached on the waist and each series to classify has just a 128 sample window of two internal sensors (a.k.a. 2.56 seconds at 50 FPS), so it amazes me how those predictions are extremely accurate given this small window of context and raw data. I've validated and re-validated that there is no important bug, and the community used and tried this code a lot. (Note: be sure to report something in the issue tab if you find bugs, otherwise [Quora](https://www.quora.com/), [StackOverflow](https://stackoverflow.com/questions/tagged/tensorflow?sort=votes&pageSize=50), and other [StackExchange](https://stackexchange.com/sites#science) sites are the places for asking questions.)

I specially did not expect such good results for guessing between the labels "SITTING" and "STANDING". Those are seemingly almost the same thing from the point of view of a device placed at waist level according to how the dataset was originally gathered. Thought, it is still possible to see a little cluster on the matrix between those classes, which drifts away just a bit from the identity. This is great.

It is also possible to see that there was a slight difficulty in doing the difference between "WALKING", "WALKING_UPSTAIRS" and "WALKING_DOWNSTAIRS". Obviously, those activities are quite similar in terms of movements. 

I also tried my code without the gyroscope, using only the 3D accelerometer's 6 features (and not changing the training hyperparameters), and got an accuracy of 87%. In general, gyroscopes consumes more power than accelerometers, so it is preferable to turn them off. 


## Improvements

In [another open-source repository of mine](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs), the accuracy is pushed up to nearly 94% using a special deep LSTM architecture which combines the concepts of bidirectional RNNs, residual connections, and stacked cells. This architecture is also tested on another similar activity dataset. It resembles the nice architecture used in "[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)", without an attention mechanism, and with just the encoder part - as a "many to one" architecture instead of a "many to many" to be adapted to the Human Activity Recognition (HAR) problem. I also worked more on the problem and came up with the [LARNN](https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network), however it's complicated for just a little gain. Thus the current, original activity recognition project is simply better to use for its outstanding simplicity. 

If you want to learn more about deep learning, I have also built a list of the learning ressources for deep learning which have revealed to be the most useful to me [here](https://github.com/guillaume-chevalier/Awesome-Deep-Learning-Resources). 


## References

The [dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) can be found on the UCI Machine Learning Repository: 

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

The RNN image for "many-to-one" is taken from Karpathy's post: 

> Andrej Karpathy, The Unreasonable Effectiveness of Recurrent Neural Networks, 2015, 
> http://karpathy.github.io/2015/05/21/rnn-effectiveness/

## Citation

Copyright (c) 2016 Guillaume Chevalier. To cite my code, you can point to the URL of the GitHub repository, for example: 

> Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016, 
> https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

My code is available for free and even for private usage for anyone under the [MIT License](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LICENSE), however I ask to cite for using the code. 

## Extra links

### Connect with me

- [LinkedIn](https://ca.linkedin.com/in/chevalierg)
- [Twitter](https://twitter.com/guillaume_che)
- [GitHub](https://github.com/guillaume-chevalier/)
- [Quora](https://www.quora.com/profile/Guillaume-Chevalier-2)
- [YouTube](https://www.youtube.com/c/GuillaumeChevalier)
- [Dev/Consulting](http://www.neuraxio.com/en/)

### Liked this project? Did it help you? Leave a [star](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/stargazers), [fork](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/network/members) and share the love!

This activity recognition project has been seen in:

- [Hacker News 1st page](https://news.ycombinator.com/item?id=13049143)
- [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow#tutorials)
- [TensorFlow World](https://github.com/astorfi/TensorFlow-World#some-useful-tutorials)
- And more.

---



```python
# Let's convert this notebook to a README automatically for the GitHub project's title page:
!jupyter nbconvert --to markdown LSTM.ipynb
!mv LSTM.md README.md
```
