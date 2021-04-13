#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -------- Code Outline -------- #
# Create a model that implements an RNN using output predictions from a CNN in order to identify rhythm changes.
# This program was used to actually train the model on the entire set. We used RR + beat label predictions as features.


# In[ ]:


import numpy as np
import pickle
import keras
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix, hamming_loss, jaccard_score
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from skmultilearn.model_selection import IterativeStratification
import matplotlib.pyplot as plt

#%%
#Directory varbiables
dir_denoised_data = 'D:/arrhythmia-database/DenoisedData/{}_de-noised.pkl'
fin_dir_peaks = 'D:/arrhythmia-database/RawDataFinal/Peaks/{}_peaks.pkl'
fin_dir_beat_labels = 'D:/arrhythmia-database/RawDataFinal/BeatLabels/{}_beat_labels.pkl'
fin_dir_rhythm_labels = 'D:/arrhythmia-database/RawDataFinal/RhythmLabels/{}_rhythm_labels.pkl'
fin_dir_rhythm_locations = 'D:/arrhythmia-database/RawDataFinal/RhythmLocations/{}_rhythm_locations.pkl'
dir_test_data = 'D:/arrhythmia-database/TestData/{}.pkl'
dir_RNN_Models = 'D:/arrhythmia-database/RNN_Models/{}'

# In[ ]:


with open(dir_denoised_data.format(1), 'rb') as f:
    y = pickle.load(f)
end = len(y)
#print(end)


# In[ ]:


# We will have time series data of labels for each patient. These series will be different lengths so we need to
# Pad them all to add 0's to the end of each sequence to get longer length inputs. So first we save how many labels are
# In each patient as well as how many beats are in the signal.

# Load in the beat labels for each patient as well as the rhythm labels for each

patient_id = range(0,48)

beat_labels = []

beat_locations = []

rhythm_labels = []

rhythm_locations = []

full_test = [14,24]
half_test = [4,28,40,41,44,45]

test_beat_labels = []
test_beat_locations = []
test_rhythm_labels = []
test_rhythm_locations = []

half_beat_labels = []
half_beat_locations = []
half_rhythm_labels = []
half_rhythm_locations = []

for i in patient_id:
    # Beat labels for patient i
    with open(fin_dir_beat_labels.format(i), 'rb') as f:
        temp_beat = pickle.load(f)
    # beat label locations for patient i
    with open(fin_dir_peaks.format(i), 'rb') as f:
        temp_beat_locations = pickle.load(f)
    # rhythm labels for patient i
    with open(fin_dir_rhythm_labels.format(i), 'rb') as f:
        temp_rhythm = pickle.load(f)
    # Rhythm label locations for patient i
    with open(fin_dir_rhythm_locations.format(i), 'rb') as f:
        temp_rhythm_location = pickle.load(f)
           
    # Split into training and testing splits
    if(i in full_test):
        test_rhythm_locations.append(temp_rhythm_location)
        test_beat_labels.append(temp_beat)
        test_rhythm_labels.append(temp_rhythm)
        test_beat_locations.append(temp_beat_locations)
    elif(i in half_test):
        half_rhythm_locations.append(temp_rhythm_location)
        half_beat_labels.append(temp_beat)
        half_rhythm_labels.append(temp_rhythm)
        half_beat_locations.append(temp_beat_locations)
    else:
        rhythm_locations.append(temp_rhythm_location)
        beat_labels.append(temp_beat)
        rhythm_labels.append(temp_rhythm)
        beat_locations.append(temp_beat_locations)


# In[ ]:


#print(len(rhythm_labels))
#print(half_rhythm_labels[-1])
#print(len(test_rhythm_labels))


# In[ ]:


# Now split the half patient test data into half in order to get representative test set
index = []
for j in half_beat_locations:
    # Find beat location nearest to the halfway point
    index.append(min(enumerate(j), key=lambda x: abs(x[1]-(end/2)))[0])
    print(index)


# In[ ]:


# Function to find the closest rhythm location to a certain beat location

def find_beat_to_rhythm(beat_location, rhythm_location):
    index, value = min(enumerate(rhythm_location), key=lambda x: abs(x[1]-(beat_location)))
    return index, value


# In[ ]:


# We have found the beat corresponding to the halfway point of the ecg, and then the function above finds the last rhythm
# Label before this point

# For every halfway beat in the list above we now need to find the nearest rhythm to it and use that as our starting rhythm

for c,k in enumerate(index):
    starting_rhythm_index, starting_rhythm_value = find_beat_to_rhythm(half_beat_locations[c][k], half_rhythm_locations[c])
    halfway_list_locations = [end / 2]
    halfway_list_labels = [half_rhythm_labels[c][starting_rhythm_index]]
    
    if(c == 0):
        # We want left half of patient ECG in the test set
        test_rhythm_labels.append(half_rhythm_labels[c][:starting_rhythm_index])
        temp_list = half_rhythm_locations[c][:starting_rhythm_index] + [end / 2]
        test_rhythm_locations.append(temp_list)
        test_beat_labels.append(half_beat_labels[c][:k])
        test_beat_locations.append(half_beat_locations[c][:k])
        # Save the right half into the training set
        temp_labels = halfway_list_labels + half_rhythm_labels[c][(starting_rhythm_index):]
        rhythm_labels.append(temp_labels)
        temp_locations = halfway_list_locations + half_rhythm_locations[c][(starting_rhythm_index):]
        rhythm_locations.append(temp_locations)
        beat_labels.append(half_beat_labels[c][k:])
        beat_locations.append(half_beat_locations[c][k:])
    else:
        # We want right half of patient ECG in the test set
        test_rhythm_labels.append(half_rhythm_labels[c][(starting_rhythm_index - 1):])
        temp_list =  [end / 2] + half_rhythm_locations[c][starting_rhythm_index:]
        test_rhythm_locations.append(temp_list)
        test_beat_labels.append(half_beat_labels[c][k:])
        test_beat_locations.append(half_beat_locations[c][k:])
        # Save the left half into the training set
        if(len(half_rhythm_labels[c]) == 1): # If we only have one rhythm present in the sample
            temp_labels = [half_rhythm_labels[c][0]]
            temp_locations = [half_rhythm_locations[c][0]]
        else:
            temp_labels = half_rhythm_labels[c][:starting_rhythm_index]
            temp_locations = half_rhythm_locations[c][:starting_rhythm_index]
        rhythm_labels.append(temp_labels)
        rhythm_locations.append(temp_locations)
        beat_labels.append(half_beat_labels[c][:k])
        beat_locations.append(half_beat_locations[c][:k])
        


# In[ ]:


# print(len(rhythm_locations))
# print(len(test_rhythm_locations))
# print(test_rhythm_labels[2])
# print(test_rhythm_locations[2])
# print(rhythm_locations[-1])
# print(rhythm_labels[-1])


# In[ ]:


#print(test_beat_labels[1])


# In[ ]:


# First we need to remove the last beat from the arrays as this is usually erroneous due to disconnection of ECG
# Can only do this for the latter half of each training set but for the first half of patients [28,40,41,44,45] we have only
# The left hand side of the ECG so the last labels etc are fine. We appended these on last so the last 5 patients do not need
# Changing

for i in beat_locations:
    del i[-1]
for j in beat_labels:   
    del j[-1]


# In[ ]:


# Check
#beat_locations[6][-1]


# In[ ]:


# Now that we have the starting and ending points of all the rhythms, we need to create windows of pre-determined
# Sample size and then these will be given a target label in the form of a vector. This vector will tell us which 
# Rhythms are present within the sample.

# Choose a sample size - The number of beat labels you are going to send in
sample_size = 10

# Now we need to create sublists for each patient with 10 beat labels in
full_list = []
for j in beat_labels:
    samples = [j[i * sample_size:(i + 1) * sample_size] for i in range((len(j) + sample_size - 1) // sample_size )]
    full_list.append(samples)


# In[ ]:


# Check
#print(len(full_list[0]))
#print(full_list[0][-1])


# In[ ]:


# Now we need to replace each type with their corresponding integer number
integer_samples = [[[0 if b == 'N'
          else 1 if b == 'L'
          else 2 if b == 'R'
          else 3 if b == 'A'
          else 4 if b == 'a'
          else 5 if b == 'J'
          else 6 if b == 'S'
          else 7 if b == 'V'
          else 8 if b == 'F'
          else 9 if b == '!'
          else 10 if b == 'e'
          else 11 if b == 'j'
          else 12 if b == 'E'
          else 13 if b == '/'
          else 14 if b == 'f'
          else 15 if b == 'x'
          else 16 if b == 'Q'
          else 17 for b in j] for j in k] for k in full_list]


# In[ ]:


# Check
#print(integer_samples[0][90])


# In[ ]:


# Check
#print(full_list[2][-1])


# In[ ]:


# Now some of the samples will be missing beats so for these we need to pad them with arbitrary values 
# To make them 10 in length

# Loop over the samples and pad the last elements
for i in integer_samples:
    for c,j in enumerate(i):
        if (c == (len(i) - 1)):
            # Pad last sample
            element_to_add = sample_size - len(j)
            for k in range(element_to_add):
                new = [99]
                j = j + new
            i[c] = j


# In[ ]:


# Check
#print(integer_samples[1][-1])


# In[ ]:


# Now we have loads of samples in integer format with the encoding from above. We now need to go through and
# Create new vector label arrays of dimension (15,) in form [1,0,0,0.....] if it is a Atrial bigeminy
# [0,1,0,0,0,0...] if it is a Atrial fibrillation etc. If the sample contains a mixture of rhythms then we opt for
# A label such as [1,1,0,0,0....] for AF and atrial bigeminy

# Now we need to create sublists for each patient with 10 beat label locations in
location_list = []
for j in beat_locations:
    samples = [j[i * sample_size:(i + 1) * sample_size] for i in range((len(j) + sample_size - 1) // sample_size )]
    location_list.append(samples)


# In[ ]:


#print(rhythm_locations[-2])
#print(location_list[-1][-1])
#print(rhythm_labels[-1])


# In[ ]:


# Now we need to loop over every sample and check where the beat labels fit into with regards to the rhythm ranges
# For ease we will assume the first few samples up the rhythm label are also the same rhythm.
# Rhythm labels:
# AB - atrial bigeminy (0), AFIB - atrial fibrillation(1), AFL - atrial flutter(2), B - ventricular bigeminy(3)
# BII - 2 heart block(4), IVR - idioventricular rhythm(5), N - normal sinus rhythm(6), NOD - nodal rhythm(7)
# P - paced rhythm(8), PREX - pre-excitation(9), SBR - sinus brachycardia(10), SVTA - supraventricular tachyarrhymia(11)
# T - ventricular trigeminy(12), VFL - ventricular flutter(13), VT - ventricular tachycardia(14)

sample_labels = []

# First pick a patient
for i,patient in enumerate(rhythm_locations):
    
    print(i)
    
    patient_beat_labels = []
    
    # Now loop over all the beat labels in that patients data
    for beat_location in beat_locations[i]:
        
        # Loop over all the rhythms and find which one it is after and use that rhythm label
        
        rhythm_after = []
        
        for c,rhythm_location in reversed(list(enumerate(patient))):
            if (len(rhythm_after) > 0):
                break
            else:
                if(beat_location > rhythm_location):
                    rhythm_after.append(rhythm_labels[i][c])
                    #print(rhythm_labels[i][c])
                
        patient_beat_labels.append(rhythm_after)
        #print(patient_beat_labels[-1])
       # print(beat_location)
        #print(i)
        
    sample_labels.append(patient_beat_labels)


# In[ ]:


#print(sample_labels[-1])
#print(len(beat_locations[8]))
#print(beat_locations[8][90:110])
#print(rhythm_locations[8])
#print(rhythm_labels[8][:10])


# In[ ]:


# Now that we have the rhythms set up we need to segment into blocks of sample_size again

full_list_labels = []
for j in sample_labels:
    samples_labels = [j[i * sample_size:(i + 1) * sample_size] for i in range((len(j) + sample_size - 1) // sample_size )]
    full_list_labels.append(samples_labels)


# In[ ]:


x = []
# Per patient
for c,i in enumerate(full_list_labels):
    print(c)
    patient = []
    # Each sample
    for j in i:
        new = []
        # Get rid of annoying list notation
        for k in j:
            try:
                new.append(k[0])
            except IndexError:
                print(k)
            #new.append(k[0])
        patient.append(new)
    x.append(patient)


# In[ ]:


#print(x[-2][-1])


# In[ ]:


#len(integer_samples[0])


# In[ ]:


# This function takes in an array of sample_labels as well as a boolean ~(tells us if there is a mixture or not)~
# We then create a corresponding vector for that sample
def create_complete_label(sample_labels):

    # Doing this finds the UNIQUE elements of the sample_labels
    unique_elements = list(set(sample_labels))
    
    #print(unique_elements)
    #for i in unique_elements:
        #print(i)
        
    # Create vector - some reason they have a bracket in front of all of them when you extract them
    label = [0 if b == 'AB'
        else 1 if b == 'AFIB'
        else 2 if b == 'AFL'
        else 3 if b == 'B'
        else 4 if b == 'BII'
        else 5 if b == 'IVR'
        else 6 if b == 'N'
        else 7 if b == 'NOD'
        else 8 if b == 'P'
        else 9 if b == 'PREX'
        else 10 if b == 'SBR'
        else 11 if b == 'SVTA'
        else 12 if b == 'T'
        else 13 if b == 'VFL'
        # This one is VT label
        else 14 for b in unique_elements]
        
    # Return the vector
    return(label)


# In[ ]:


complete_labels = []
for i in x:
    patient_l = []
    for j in i:
        patient_l.append(create_complete_label(j))
        
    complete_labels.append(patient_l)


# In[ ]:


#print(complete_labels[-1])
#print(rhythm_labels[7])


# In[ ]:


# Now we need to transfer these into vectors like [1,0,0,0,0,0,0,0,0,0...] if label 1 is present or
# [1,1,0,0,0,0,0,0,0....] if a mixture of label 1 or 2 are present etc.

y = []

# Loop over every sample and create a vector for it and append it to a list
for c,patient in enumerate(complete_labels):
    
    for sample in patient:
        
        # Create a label vector
        temp = np.zeros((15,))
        
        # Find which numbers are present, then these indices need to be set to 1
        for item in sample:
            temp[item] = 1
            
        y.append(temp)


# In[ ]:


# Have complete set of target labels and input arrays
# Check
#print(integer_samples[0])
# Need to convert integer_samples to a 1 dimensional array of 25 sample windows
#print(len(y))


# In[ ]:


x = []
for i in integer_samples:
    for j in i:
        x.append(j)


# In[ ]:


x = np.array(x)


# In[ ]:


x.flatten()


# In[ ]:


x = x.reshape(len(y), sample_size,1)
#print(x.shape)


# In[ ]:


#x[400]


# In[ ]:


# -------- SECTION FOR RR INTERVAL ADDITION -------- #
# See what effect adding RR interval lengths will do if we use it as a second feature.
# Each peak will take a value for its RR length to then next peak, except the last one which will just
# Be set to 0

# First split each patients beats into chunks of sample size
patient_sizes = []
new_locations = []
for k in range(len(beat_locations)):
    
    patient_sizes.append(len(beat_locations[k]) / sample_size)
    
    new_locations.append([beat_locations[k][i:i + sample_size] for i in range(0, len(beat_locations[k]), sample_size)])


# In[ ]:


# Now find the distance between all the R peaks within that window and set the very final one to 0 so it is same length
def distance(point1, point2):
    return(abs(point1 - point2))

RR_distances = []
for patient in new_locations:
    patient_RR = []
    for c,sample in enumerate(patient):
        temp = []
        
        if (c != len(patient) - 1):
            for i in range(0, sample_size, 1):
                if(i == sample_size - 1):
                    temp += [0]
                else:    
                    temp += [distance(sample[i],sample[i+1])]
        
        # Last set for patient will not be a complete set of sample size so work out what is in there to do then pad with 0's
        else:
        
            number_peaks = len(sample)
            
            number_to_pad = sample_size - number_peaks
            
            for i in range(0, number_peaks, 1):
                if(i == number_peaks - 1):
                    temp += [0]
                else:
                    temp += [distance(sample[i],sample[i+1])]
                    
            for j in range(number_to_pad):
                temp += [0]
        
        patient_RR.append(temp)
    RR_distances += patient_RR


# In[ ]:


RR_distances = np.array(RR_distances).reshape((x.shape))


# In[ ]:


#print(RR_distances.shape)
#print(x.shape)


# In[ ]:


#print(RR_distances[0])
#print(x[0])


# In[ ]:


# Now we have two arrays that we need to join together to get 2 feature arrays
total_x = np.concatenate((x,RR_distances), axis = 2)
#print(total_x.shape)
#print(total_x[0])


# In[ ]:


y = np.array(y)
#print(y.shape)


# In[ ]:


#print(y[170])


# In[ ]:


# Now define a function that finds proportions of the different labels

def proportions(labels):
    
    # Data is in vector form so [1,0,0,0,0....] etc with this indicating it is the first rhythm
    # We need to loop over the vector to find where the 1's are then we can get the proportion that is a certain type
    
    n_mixed, n_pure, n_AB, n_AFIB, n_AFL, n_B, n_BII, n_IVR, n_N, n_NOD, n_P, n_PREX, n_SBR, n_SVTA, n_T, n_VFL, n_VT = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    
    options = {0 : n_AB,
           1 : n_AFIB,
           2 : n_AFL,
           3 : n_B,
           4 : n_BII,
           5 : n_IVR,
           6 : n_N,
           7 : n_NOD,
           8 : n_P,
           9 : n_PREX,
          10 : n_SBR,
          11 : n_SVTA,
          12 : n_T,
          13 : n_VFL,
          14 : n_VT,
          15 : n_mixed,
          16 : n_pure
}

    for i in labels:
        #print(i)
        mix = False
        count = 0
        for index,number in enumerate(i):
            #print(count)
            #print(number)
            
            #if ((number == 1) and (index == 0)):
                #print(i)
            
            if(number == 1):
                count += 1
            
                # Change variable in dictionary to one up
                options[index] += 1
                
            elif(number == 0):
                count = count
        
        if(count >= 2):
                mix = True
                options[15] += 1
        elif(count == 1):
                mix = False
                options[16] += 1
                
    return options


# In[ ]:


data_split = proportions(y)


# In[ ]:


#print(data_split)

# This checks we have correct number of samples
#print(data_split[15] + data_split[16])


# In[ ]:


# Save the training data and training labels
#with open('RNN Training Data.pkl', 'wb') as f:
 #   pickle.dump(x, f)
#with open('RNN Training Targets.pkl', 'wb') as f:
 #   pickle.dump(y, f)


# In[ ]:


import math

batch_size = 28


# In[ ]:


# Now save the indices for GPU use on server
#with open('RNN Training Indices.pkl', 'wb') as f:
 #   pickle.dump(train_indices, f)
#with open('RNN Validation Indices.pkl', 'wb') as f:
 #   pickle.dump(val_indices, f)
#%%
def recall_m(y, x):
    true_positives = K.sum(K.round(K.clip(y * x, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y, x):
    true_positives = K.sum(K.round(K.clip(y * x, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(x, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y, x):
    precision = precision_m(y, x)
    recall = recall_m(y, x)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# In[ ]:


from keras import optimizers

optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Create the model
model = Sequential()
model.add(Bidirectional(LSTM(units = 256,
                            stateful = False,
                            recurrent_dropout = 0.2,
                            activation = 'sigmoid'),
                            input_shape = (sample_size,2)))

model.add(Dense(15, activation = 'sigmoid'))

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',keras.metrics.Precision(),recall_m,f1_m,])
    
# Fit to ALL the data
    
history = model.fit(total_x, y, verbose = 1, batch_size = batch_size, epochs = 50, shuffle = True)

# Save the completed model
model.save(dir_RNN_Models.format('RNN_Model.h5'))

#%%
matrix = multilabel_confusion_matrix(y.argmax(axis=1), x.argmax(axis=1))
print(matrix)