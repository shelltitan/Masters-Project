#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This piece of code implements the full testing portion from CNN -> RNN -> output


import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras
import pickle
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('P:/')
# The generator needed depends on the model. For Hengguis CNN use the first, for transfer learning use the second
from Arrhythmia_generator import DataGenerator
#from VGG_arrhythmia import DataGenerator


# In[5]:


# Parameters
params = {'dim': (128,128),
          'batch_size': 1,
          'n_classes': 10,
          'n_channels': 2,
          'shuffle': True}


# In[3]:

# Load in the model 
CNN = load_model('HengguiCNN/CNN_less_classes_less_norm_oversampling_no_aug.h5')
print('This is the CNN_less_classes_less_norm_oversampling_no_aug.h5 model.')

# In[6]:





# In[4]:




# In[6]:




    # Need to load in the data that is going to be used for testing 
with open('Test Data/Test IDs Less Classes.pkl', 'rb') as f:
    ids = pickle.load(f)
with open('Test Data/Test Labels Less Classes OHE.pkl', 'rb') as f:
    labels = pickle.load(f)

# This gets the predictions from the model on the test set
training_generator = DataGenerator(ids, labels , **params)
#patient_features = mobile(ids,labels)
flat_predictions = CNN.predict_generator(training_generator,verbose = True)
with open('Test Data/Test Labels Less Classes OHE.pkl', 'rb') as f:
    true_labels = pickle.load(f)
flat_true = np.array(true_labels)


# In[13]:



# In[14]:


maxes = np.argmax(flat_predictions, axis = 1)
for c,i in enumerate(maxes):
    flat_predictions[c] = np.zeros((10,))
    flat_predictions[c][i] = 1
#flat_predictions[flat_predictions != np.amax(flat_predictions, axis = 1)] = 0
# Do a count check to make sure only one in per label
tester = list(flat_predictions)
for i in tester:
    if (list(i).count(1) > 1):
        print(i)


# In[15]:


print(flat_predictions.shape)
print(flat_true.shape)


# In[16]:


# Now that we have the one hot encoded categories lets find the confusion matrix
print(metrics.multilabel_confusion_matrix(flat_true, flat_predictions))


# In[17]:
# To get the full matrix, use this:
cat_true = []
for i in flat_true:
    cat_true.append(np.argmax(i))
cat_predictions = []
for i in flat_predictions:
    cat_predictions.append(np.argmax(i))
confusion_matrix = metrics.confusion_matrix(cat_true, cat_predictions,normalize='true')
print(confusion_matrix)

#with open('FINAL MODEL CONFUSION MATRIX MICRO.pkl', 'wb') as f:
#   pickle.dump(confusion_matrix,f)

# Find Precision, Recall and F1 scores 
from sklearn.metrics import f1_score, accuracy_score, precision_score,recall_score
print(precision_score(flat_true, flat_predictions, average = 'macro'))
print(recall_score(flat_true, flat_predictions, average = 'macro'))
print(f1_score(flat_true, flat_predictions, average = None))
print(f1_score(flat_true, flat_predictions, average = 'macro'))
print(accuracy_score(flat_true, flat_predictions))


# In[ ]:




