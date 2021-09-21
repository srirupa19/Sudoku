import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
import imutils

# matplotlib inline
sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label']
X_train = train.drop(labels = 'label' , axis = 1)
del train

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape data
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Make Y categorical data
y_train = to_categorical(y_train, num_classes = 10)

m = X_train.shape[0]

# Add rotated data
shape = (X_train.shape[0] , 28 , 28 )
new_X_train1 = np.zeros(shape , dtype=float)
new_X_train2 = np.zeros(shape , dtype=float)

for i in range(m):
	new_X_train1[i] = imutils.rotate(X_train[i], 10)
	new_X_train2[i] = imutils.rotate(X_train[i], 350)

#new_X_train.reshape(m , 28 , 28 , 1)
X_train = np.concatenate((X_train, new_X_train1.reshape(m , 28 , 28 , 1) , new_X_train2.reshape(m , 28 , 28 , 1) ))
y_train = np.concatenate((y_train , y_train , y_train))
del new_X_train1 , new_X_train2
#X_train.shape

# Split the train and the validation set for the fitting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)

# VGG
model = Sequential()
# Make images ( 32 , 32 )
model.add(ZeroPadding2D(2))
# 6 ( 5 , 5 ) filters , stride = 1
model.add(Conv2D(filters = 32 , kernel_size = (3,3), strides=(1 , 1) , activation='relu', padding='Same'))
# Avg pool f = 2 , s = 2
model.add(MaxPool2D(pool_size=(2 , 2),strides=(2 , 2)))
#model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# 6 ( 5 , 5 ) filters , stride = 1
model.add(Conv2D(filters = 64 , kernel_size = (3,3), strides=(1 , 1) , activation='relu', padding='Same'))
# Avg pool f = 2 , s = 2
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# 6 ( 5 , 5 ) filters , stride = 1
model.add(Conv2D(filters = 128 , kernel_size = (3,3), strides=(1 , 1) , activation='relu', padding='Same'))
# Avg pool f = 2 , s = 2
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
# 84 FC
model.add(Dense(1024, activation = "relu" ))
model.add(Dense(1024, activation = "relu" ))

model.add(Dense(10, activation = "softmax"))

# Training
opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer = opt , loss = "categorical_crossentropy", metrics=["accuracy"])

"""learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)"""
model.fit(X_train, y_train, batch_size = 64, epochs = 30, validation_data = (X_val, y_val), verbose = 2)




model.save('/home/sriru/Srirupa/Sudoku/model_1')

