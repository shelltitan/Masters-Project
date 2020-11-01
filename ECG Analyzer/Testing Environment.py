#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This piece of code implements the full testing portion from CNN -> RNN -> output


# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import pickle
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from keras.applications import MobileNet
import sys
sys.path.append('P:/')
# The generator needed depends on the model. For custom CNN use the first, for transfer learning use the second
from Arrhythmia_generator import DataGenerator
#from VGG_arrhythmia import DataGenerator


# In[ ]:


# Parameters
params = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 10,
          'n_channels': 2,
          'shuffle': False}


# In[ ]:


# Bring in the RNN model as well as the CNN model
RNN = load_model('RNN Models/RNN Model.h5')

# Change model depending on which we are using
CNN = load_model('CNN Models/CNN_less_classes_less_norm_oversampling_no_aug.h5')


# In[ ]:


CNN.summary()


# In[ ]:


# For the transfer learning models I have to send the input images through the conv_base first
def mobile(X_train,Y_train):
    conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))
    samples_generator = DataGenerator(X_train, Y_train , **params)
    test_features = conv_base.predict_generator(samples_generator, verbose=True)
    
    return test_features


# In[ ]:


# Check the patient labels and that
   # with open('Teddys Stuff/Test Data/Patient {} IDs.pkl'.format(2), 'rb') as f:
    #    ids = pickle.load(f)
   # with open('Teddys Stuff/Test Data/Patient {} Labels.pkl'.format(2), 'rb') as f:
    #    labels = pickle.load(f)


# In[ ]:


# This loops over every patient and gets the predictions of the beat type
predictions = []
true_labels = []
patient_indexes = []
#for i in range(8):
# Need to load in the data that is going to be used for testing 
with open('Test Data/Test IDs Less Classes.pkl', 'rb') as f:
    ids = pickle.load(f)
with open('Test Data/Test Labels Less Classes OHE.pkl', 'rb') as f:
    labels = pickle.load(f)
        
        
training_generator = DataGenerator(ids, labels , **params)
#patient_features = mobile(ids,labels)
patient_prediction = CNN.predict_generator(training_generator,verbose = True)
    
    
#predictions.append(patient_prediction)
#true_labels.append(labels)
#print(len(labels))
    #if (i==0):
   #     patient_indexes.append(len(labels))
    #else:
   #     patient_indexes.append(len(labels) + patient_indexes[i - 1])
   # print(patient_indexes[i])


# In[ ]:


#print(labels[0])


# In[ ]:


#print(patient_prediction[0])


# In[ ]:


# Make a copy so we do not mess up the predictions and have to run it again
copy = patient_prediction.copy()


# In[ ]:


#print(copy[0])


# In[ ]:


# Put ones/zeros at the max element etc
copy = (copy == copy.max(axis=1)[:,None]).astype(int)


# In[ ]:


#print(copy[0])
#print(copy.shape)


# In[ ]:


#print(copy[3])

# Observe all the metrics
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
print(accuracy_score(labels,copy))
print(precision_score(labels,copy, average = None))
print(recall_score(labels,copy, average = None))
print(f1_score(labels,copy, average = None))
print(precision_score(labels,copy, average = 'micro'))
print(recall_score(labels,copy, average = 'micro'))
print(f1_score(labels,copy, average = 'micro'))
print(multilabel_confusion_matrix(labels,copy))


# In[ ]:


# Now we need to convert these into labels for the RNN so we need to put these x's into labels first
# Wherever the 1 is elementwise, we need to save this as its label
RNN_x = []

for i in copy:
    for c,element in enumerate(i):
        if(element == 1):
            RNN_x.append(c)


# In[ ]:


#print(RNN_x[:10])
RNN_x = np.array(RNN_x)
#print(RNN_x.shape)


# In[ ]:


# now split this array into 10 window chunks
rnn_x = [RNN_x[i:i + 10] for i in range(0, len(RNN_x), 10)]


# In[ ]:


rnn_x = np.array(rnn_x)
#print(rnn_x.shape)


# In[ ]:


# 772 onwards are ! beats only and these all belong to VFL rhythm which we cannot detect therefore remove these
rnn_x = list(rnn_x)
del rnn_x[772:]
rnn_x = np.array(rnn_x)
#print(rnn_x.shape)


# In[ ]:


#print(rnn_x[:2])


# In[ ]:


# Bring in the correct labels as well as the test beat locations in order to find RR intervals

with open('RNN Models/RNN_Test_y.pkl', 'rb') as f:
    rnn_true_y = pickle.load(f)
with open('RNN Models/RNN_Test_Beat_Locations.pkl', 'rb') as f:
    beat_locations = pickle.load(f)


# In[ ]:


#print(beat_locations[-1])


# In[ ]:


# Now find the RR intervals and stack them with the beat label array
def distance(point1, point2):
    return(abs(point1 - point2))

RR_intervals = []
for c,sample in enumerate(beat_locations):
    temp = []
        
    for i in range(0, 10, 1):
        if(i == 9):
            temp += [0]
        else:    
            temp += [distance(sample[i],sample[i+1])]
                
    RR_intervals.append(temp)


# In[ ]:


RR_intervals = np.array(RR_intervals)
RR_intervals = RR_intervals.reshape((RR_intervals.shape[0], RR_intervals.shape[1], 1))
rnn_x = rnn_x.reshape((RR_intervals.shape))
#print(RR_intervals.shape)
#print(RR_intervals[-3])


# In[ ]:


cnn_output_x = np.concatenate((rnn_x, RR_intervals), axis = 2)
#print(cnn_output_x.shape)


# In[ ]:


# Now use these to detect the rhythms
rhythm_preds = RNN.predict(cnn_output_x, verbose = True)


# In[ ]:


#print(rhythm_preds[0])
# Again round the probabilities appropriately
rhythm_preds[rhythm_preds > 0.5] = 1
rhythm_preds[rhythm_preds <= 0.5] = 0
#print(rhythm_preds[0])
#print(rnn_true_y[0])


# In[ ]:


# Now find confusion etc
print(accuracy_score(rnn_true_y,rhythm_preds))
print(precision_score(rnn_true_y,rhythm_preds, average = None))
print(recall_score(rnn_true_y,rhythm_preds, average = None))
print(f1_score(rnn_true_y,rhythm_preds, average = None))
print(precision_score(rnn_true_y,rhythm_preds, average = 'micro'))
print(recall_score(rnn_true_y,rhythm_preds, average = 'micro'))
print(f1_score(rnn_true_y,rhythm_preds, average = 'micro'))
print(multilabel_confusion_matrix(rnn_true_y,rhythm_preds))


# In[ ]:


# Lets try and see what difference removing the mixed labels makes and just treating the rhythms as multiclass problem instead
pure_x = []
pure_y = []
mixed_x = []
mixed_y = []

for c,label in enumerate(rnn_true_y):
    # If it is pure then use it
    if(list(label).count(1) == 1):
        pure_x.append(cnn_output_x[c])
        pure_y.append(label)
        
    else:
        mixed_x.append(cnn_output_x[c])
        mixed_y.append(label)


# In[ ]:


pure_x = np.array(pure_x)
pure_y = np.array(pure_y)
#print(pure_x.shape)
#print(pure_y.shape)

mixed_x = np.array(mixed_x)
mixed_y = np.array(mixed_y)
#print(mixed_x.shape)
#print(mixed_y.shape)


# In[ ]:


# Now send through RNN and see results
Multiclass_RNN = load_model('RNN Models/Multiclass.h5')

mixed_preds = RNN.predict(mixed_x,verbose = True)
mixed_preds[mixed_preds > 0.5] = 1
mixed_preds[mixed_preds <= 0.5] = 0

pure_preds = Multiclass_RNN.predict(pure_x, verbose = True)
pure_preds[pure_preds > 0.5] = 1
pure_preds[pure_preds <= 0.5] = 0


# In[ ]:


print(accuracy_score(mixed_y,mixed_preds))
print(precision_score(mixed_y,mixed_preds, average = None))
print(recall_score(mixed_y,mixed_preds, average = None))
print(f1_score(mixed_y,mixed_preds, average = None))
print(precision_score(mixed_y,mixed_preds, average = 'micro'))
print(recall_score(mixed_y,mixed_preds, average = 'micro'))
print(f1_score(mixed_y,mixed_preds, average = 'micro'))
print(multilabel_confusion_matrix(mixed_y,mixed_preds))


# In[ ]:


print(accuracy_score(pure_y,pure_preds))
print(precision_score(pure_y,pure_preds, average = None))
print(recall_score(pure_y,pure_preds, average = None))
print(f1_score(pure_y,pure_preds, average = None))
print(np.mean(precision_score(pure_y,pure_preds, average = None)))
print(np.mean(recall_score(pure_y,pure_preds, average = None)))
print(np.mean(f1_score(pure_y,pure_preds, average = None)))
#print(multilabel_confusion_matrix(pure_y,pure_preds))


# In[ ]:


#print((0.87356322 + 0.96969697 + 0.90501601 +  0.64705882 + 0.96428571 + 1 + 0)/7)

