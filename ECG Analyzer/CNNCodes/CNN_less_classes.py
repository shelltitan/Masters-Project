#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import pickle
from keras.models import Sequential
import sys
sys.path.append('D:/')
from Arrhythmia_generator import DataGenerator
#from Arrhythmia_generator_aug_less_classes import DataGenerator_aug
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score

# In[ ]:


with open('Segmented Data/New Inputs Less Classes/TrainVal Ids less.pkl', 'rb') as f:
    Trainval_ids = pickle.load(f)
with open('Segmented Data/New Inputs Less Classes/TrainVal labels OHE less.pkl', 'rb') as f:
    Trainval_labels = pickle.load(f)
    
# I need to open the indices for the different folds as well
with open('Segmented Data/New Inputs Less Classes/Train indices aug.pkl', 'rb') as f:
    Train_indices = pickle.load(f)
    
with open('Segmented Data/New Inputs Less Classes/Val indices.pkl', 'rb') as f:
    Val_indices = pickle.load(f)
    
with open('Segmented Data/New Inputs Less Classes/Final train indices aug.pkl', 'rb') as f:
    Final_train_indices = pickle.load(f)    
   

print('This is training the full training set for undersampling of normal samples and oversampling but with no aug')


# In[ ]:

def proportions(labels):
    
    # Data is in vector form so [1,0,0,0,0....] etc with this indicating it is the first rhythm
    # We need to loop over the vector to find where the 1's are then we can get the proportion that is a certain type
    
    n_N, n_L, n_R, n_A, n_a, n_J, n_S, n_V, n_F, n_exchlaim, n_e, n_j, n_E, n_bachslach, n_f, n_x, n_Q, n_line = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    
    options = {0 :n_N,
           1 : n_L,
           2 : n_R,
           3 : n_A,
           4 : n_a,
           5 : n_J,
           6 : n_S,
           7 : n_V,
           8 : n_F,
           9 : n_exchlaim,
          10 : n_e,
          11 : n_j,
          12 : n_E,
          13 : n_bachslach,
          14 : n_f,
          15 : n_x,
          16 : n_Q,
          17 : n_line
}

    for i in labels:
        #print(i)
        for index,number in enumerate(i):
            #print(count)
            #print(number)
            
            #if ((number == 1) and (index == 0)):
                #print(i)
            
            if(number == 1):
                # Change variable in dictionary to one up
                options[index] += 1
        
                
    return options

print(proportions(Trainval_labels))
samp = [Trainval_labels[j] for j in Train_indices[0]]
print(proportions(samp))
# Parameters
params = {'dim': (128,128),
          'batch_size': 25,
          'n_classes': 10,
          'n_channels': 2,
          'shuffle': True}

params_val = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 10,
          'n_channels': 2,
          'shuffle': True}
# In[ ]:


# Implement a sequential network
def createModel():
    conv_window = (10, 10)
    model = Sequential()
    model.add(Conv2D(32, (10, 10),  activation='relu', input_shape=(128, 128, 2))),
    model.add(Conv2D(32, (10, 10), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5)) # Weight decay rate? 1E-6
    model.add(Conv2D(32, (8, 8), activation = 'relu'))
    model.add(Conv2D(32, (4, 4), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5)),
    model.add(Dense(10, activation='softmax')) #softmax?
    sgd = optimizers.SGD(lr=0.001, decay=0.000001, momentum=0.8, nesterov=True)
    model.compile(optimizer=sgd, loss ='categorical_crossentropy', metrics=['mse', 'mae', 'categorical_accuracy'])
    return model


# In[ ]:
X_final = [Trainval_ids[j] for j in Final_train_indices]
Y_final = [Trainval_labels[j] for j in Final_train_indices]
training_generator = DataGenerator(X_final, Y_final , **params)
model = createModel()
history = model.fit_generator(generator = training_generator,epochs=30,verbose=True)

model.save('HengguiCNN/CNN_less_classes_less_norm_oversampling_no_aug.h5')
with open('HengguiCNN/History_CNN_less_classes_less_norm_oversampling_no_aug.pkl', 'wb') as f:
    pickle.dump(history.history,f)


f1_tot = np.zeros((10,))
history_list = []
for i in range(len(Train_indices)):
    print('Epoch number:', i)
    # Need to get the correct samples for the training and val sets
 
    X_train = [Trainval_ids[j] for j in Train_indices[i]]
    Y_train = [Trainval_labels[j] for j in Train_indices[i]]
    X_val = [Trainval_ids[j] for j in Val_indices[i]]
    Y_val = [Trainval_labels[j] for j in Val_indices[i]]

    training_generator = DataGenerator(X_train, Y_train , **params)
    print('done1')
    validation_generator = DataGenerator(X_val, Y_val, **params)
    print('done2')
    
    model=createModel()
    history = model.fit_generator(generator = training_generator, validation_data = validation_generator,epochs=20,verbose=True)
    history_list.append(history.history)
    
    scores_generator = DataGenerator(X_val, Y_val , **params_val)
    #patient_features = mobile(ids,labels)
    flat_predictions = model.predict_generator(scores_generator,verbose = True)
    flat_true = np.array(Y_val)
    
    maxes = np.argmax(flat_predictions, axis = 1)
    for c,i in enumerate(maxes):
        flat_predictions[c] = np.zeros((10,))
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
    f1 = f1_score(flat_true, flat_predictions, average = None)
    f1_tot = f1_tot + f1
    # print(f1_tot)
    #print(f1.shape)
    
    print(accuracy_score(flat_true, flat_predictions))
    
    
with open('HengguiCNN/CNN_history_list_less_classes_less_norm_oversampling_no_aug.pkl', 'wb') as f:
    pickle.dump(history_list,f)
f1_av = f1_tot / 10
print(f1_av)
tot_f1_av = np.sum(f1_av) / 10
print('The total F1 score average is: ' ,tot_f1_av)

# In[ ]:

X_final = [Trainval_ids[j] for j in Final_train_indices]
Y_final = [Trainval_labels[j] for j in Final_train_indices]
training_generator = DataGenerator(X_final, Y_final , **params)
model = createModel()
history = model.fit_generator(generator = training_generator,epochs=20,verbose=True)

model.save('HengguiCNN/CNN_less_classes_less_norm_oversampling_no_aug.h5')
with open('HengguiCNN/History_CNN_less_classes_less_norm_oversampling_no_aug.pkl', 'wb') as f:
    pickle.dump(history.history,f)



