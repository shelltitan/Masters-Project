#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import pickle
from keras.models import Sequential
import sys
sys.path.append('P:/')
from VGG_arrhythmia import DataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.applications import MobileNet
from keras import models
from keras import layers
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
# In[2]:


conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))


# In[ ]:


with open('Segmented Data/Final Inputs/Trainval ids sampleless.pkl', 'rb') as f:
    Trainval_ids = pickle.load(f)
with open('Segmented Data/Final Inputs/TrainVal Labels OHE.pkl', 'rb') as f:
    Trainval_labels = pickle.load(f)
print(len(Trainval_ids))    
# I need to open the indices for the different folds as well
with open('Segmented Data/Final Inputs/Train indices.pkl', 'rb') as f:
    Train_indices = pickle.load(f)
with open('Segmented Data/Final Inputs/Val indices.pkl', 'rb') as f:
    Val_indices = pickle.load(f)

# In[ ]:


# I need to get all the input features first before putting it into the dense NN.

# Parameters
params = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 18,
          'n_channels': 3,
          'shuffle': True}

#trainval_generator = DataGenerator(Trainval_ids, Trainval_labels , **params)

#conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(128,128,3))
#Trainval_features = conv_base.predict_generator(trainval_generator, verbose=True)


# In[ ]:


def create_model():  
    model = Sequential()
    model.add(Flatten(input_shape=(4,4,1024)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax'))
    
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss ='categorical_crossentropy', metrics=['mse', 'mae', 'categorical_accuracy'])

    return model


# In[ ]:

# Now I can do the K-fold loop
history_list= []
for i in range(len(Train_indices)):
    print(i)
    print('Done1')
    # Need to get the correct samples for the training and val sets
    X_train = [Trainval_ids[j] for j in Train_indices[i]]
    Y_train = [Trainval_labels[j] for j in Train_indices[i]]
    X_val = [Trainval_ids[j] for j in Val_indices[i]]
    Y_val = [Trainval_labels[j] for j in Val_indices[i]]
    print('Done2')
    
    
    
    
    conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))
    train_generator = DataGenerator(X_train, Y_train , **params)
    X_train = conv_base.predict_generator(train_generator, verbose=True)
    print('Done3')
    val_generator = DataGenerator(X_val, Y_val , **params)
    X_val = conv_base.predict_generator(val_generator, verbose=True)
    
    
    
    Y_train = [Trainval_labels[j] for j in Train_indices[i]]
    Y_val = [Trainval_labels[j] for j in Val_indices[i]]
    # Need to have the same amount of labels as X_train and X_val, as theyre done in batches of 25:
    rem1 = len(Y_train) % 25
    if not rem1 == 0:
        Y_train = Y_train[:-rem1][:]
    rem2 = len(Y_val) % 25
    if not rem2 == 0:
        Y_val = Y_val[:-rem2][:]
    
    
    print(X_train.shape)
    print(len(Y_train))
    print(len(Y_train[0]))
    
    
    # Just in case
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    
    
    
    model=create_model()
    model.summary()
    history = model.fit(x = X_train, y = Y_train,validation_data = (X_val,Y_val),epochs=10,verbose=True)
    history_list.append(history.history)
    
    
    #patient_features = mobile(ids,labels)
    flat_predictions = model.predict(X_val,verbose = True)
    flat_true = np.array(Y_val)
    
    maxes = np.argmax(flat_predictions, axis = 1)
    for c,i in enumerate(maxes):
        flat_predictions[c] = np.zeros((18,))
        flat_predictions[c][i] = 1
    
    print(flat_predictions.shape)
    print(flat_true.shape)
    print(metrics.multilabel_confusion_matrix(flat_true, flat_predictions))
    
    cat_true = []
    for i in flat_true:
        cat_true.append(np.argmax(i))
    cat_predictions = []
    for i in flat_predictions:
        cat_predictions.append(np.argmax(i))
    confusion_matrix = metrics.confusion_matrix(cat_true, cat_predictions,normalize='true')
    print(confusion_matrix)
    
    print(f1_score(flat_true, flat_predictions, average = None))
    print(accuracy_score(flat_true, flat_predictions))
    
    
    
#with open('MobileNet_history_list.pkl', 'wb') as f:
    #pickle.dump(history_list,f)


# In[ ]:


# Now I can train on all the data 

params2 = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 18,
          'n_channels': 3,
          'shuffle': True}

conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))
train_generator = DataGenerator(Trainval_ids, Trainval_labels , **params2)
Trainval_features = conv_base.predict_generator(train_generator, verbose=True)

model = create_model()
history = model.fit(Trainval_features,Trainval_labels,epochs = 25,verbose=True)

#model.save('MobileNet_basic.h5')
#with open('History_MobileNet.pkl', 'wb') as f:
    #pickle.dump(history.history,f)


# In[ ]:





# In[ ]:




