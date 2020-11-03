#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This code trains the entire R Peak Detection algorithm as well as
# Test it on a hold out set of three patients that will contain a mix of ECG
# Signals. This will allow us to see how well it will perform in practise.


# In[ ]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
# For the LSTM
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, GlobalMaxPool1D, Dense, Dropout, Flatten
# For data splitting
from sklearn.model_selection import train_test_split
# Potentially for analysing probability output distribution
from scipy.signal import find_peaks
from scipy import interpolate

#%%
#Directory Variables
dir_denoised_data = 'D:/arrhythmia-database/DenoisedData/{}_de-noised.pkl'
rpeak_dir = 'D:/arrhythmia-database/RPeakDetector'
fin_dir_peaks = 'D:/arrhythmia-database/RawDataFinal/Peaks/{}_peaks.pkl'


# In[ ]:


# Use this function to normalise the ECG signals

def normalise(x):
    mean = np.mean(x)
    x -= mean
    std = np.std(x)
    x /= std
    return (x)


# In[ ]:


# Test set is full of patients: 100, 105, 116, 215, 232
# All the other patients can be put into the training set

train = [1,3,5,7,9,11,13,17,19,21,23,25,27,29,31,33,35,37,39,41,43]

final_x = np.zeros((len(train), 65,10000,6), dtype=np.float32)
final_y = np.zeros((len(train), 65, 10000), dtype=np.int32)

# Loop over patients and create an appropriate input and label array
for o,i in enumerate(train):
    
    #print(i)
    
    # Load in the raw ECG as well as the clinician annotated peaks
    with open(dir_denoised_data.format(i), 'rb') as f:
        smoothed = pickle.load(f)
    
    with open(fin_dir_peaks.format(i), 'rb') as f:
        peaks = pickle.load(f)
        
    # We have two leads so extract them first
        
    lead_1 = np.zeros((smoothed.shape[0],))
    lead_2 = np.zeros((smoothed.shape[0],))
    
    for c,dual_point in enumerate(smoothed):
        lead_1[c] = dual_point[0]
        lead_2[c] = dual_point[1]
        
    # We are going to implement chunks of 10000 samples
    time_stamp = 10000
    
    detrend_lead_1 = np.asarray([lead_1[i: i + time_stamp] for i in range(0, len(lead_1), time_stamp)])
    detrend_lead_2 = np.asarray([lead_2[i: i + time_stamp] for i in range(0, len(lead_2), time_stamp)])
    
    patient_x = np.zeros((65,10000,6), dtype=np.float32)
    
    # Use these knots to fit the cubic splines
    knot = np.arange(10,10000, 100)
    
    for index,chunk in enumerate(detrend_lead_1):
        
        # First we detrend the denoised ECG signal in order to remove baseline wander
        # Do this via fitting a cubic spline then removing this from the original signal
        tck_1 = interpolate.splrep(list(range(time_stamp)), detrend_lead_1[index],t=knot, k = 3, task = -1)
        new_lead_1 = np.array(np.array(detrend_lead_1[index]) - np.array(interpolate.splev(list(range(time_stamp)), tck_1)), dtype = np.float32)
        tck_2 = interpolate.splrep(list(range(time_stamp)), detrend_lead_2[index],t=knot, k = 3, task = -1)
        new_lead_2 = np.array(np.array(detrend_lead_2[index]) - np.array(interpolate.splev(list(range(time_stamp)), tck_2)), dtype = np.float32)
        smoothed = np.array(normalise(new_lead_1),dtype=np.float32)
        smoothed2 = np.array(normalise(new_lead_2),dtype=np.float32)
        
        # Calculate the derivative and second order derivatives
        
        deriv = np.array(np.gradient(smoothed), dtype=np.float32)
        deriv = np.array(normalise(deriv),dtype=np.float32)
        deriv2 = np.array(np.gradient(smoothed2),dtype=np.float32)
        deriv2 = np.array(normalise(deriv2),dtype=np.float32)
        deriv_2 = np.array(np.gradient(deriv),dtype=np.float32)
        deriv_2 = np.array(normalise(deriv_2),dtype=np.float32)
        deriv2_2 = np.array(np.gradient(deriv2),dtype=np.float32)
        deriv2_2 = np.array(normalise(deriv2_2),dtype=np.float32)
        
        # Stack all these features together to get a complete input array
        x = np.hstack((smoothed,deriv))
        x = np.hstack((x,deriv_2))
        x = np.hstack((x,smoothed2))
        x = np.hstack((x,deriv2))
        x = np.hstack((x,deriv2_2))

        new_x = np.zeros((10000, 6),dtype=np.float32)

        for l in range(smoothed.shape[0]):
            temp = np.array([smoothed[l], deriv[l], deriv_2[l], smoothed2[l], deriv2[l], deriv2_2[l]], dtype=np.float32)
            new_x[l] = temp
            
        patient_x[index] = new_x
        
    y = np.zeros((len(lead_1),))
    
    # Define a buffer either side of peak to allow this as an R peak location
    # This way we have a very small 'region' of R peak allowed locations
    buffer = 2
    
    # Set the labels to true in buffer region
    for k in peaks:
        y[k - buffer: k + buffer] = 1
    
    patient_y = np.asarray([y[i: i + time_stamp] for i in range(0, len(y), time_stamp)], dtype=np.int32)
    
    # Now append these to the total
    final_x[o] = patient_x
    final_y[o] = patient_y


# In[ ]:


#print(len(final_x))
#print(len(final_x[0]))


# In[ ]:


final_x = np.array(final_x)


# In[ ]:


#print(final_x.shape)


# In[ ]:


#print(final_x[1][0])


# In[ ]:


# Reshape so that we just have all the points with 6 features

total = (final_x.shape[0] * final_x.shape[1] * final_x.shape[2])
test_x = np.reshape(final_x, (total, final_x.shape[-1]))


# In[ ]:


#test_x.shape


# In[ ]:


#test_x[650000]
# Re-shaping has worked so now do same for labels


# In[ ]:


final_y = np.array(final_y)


# In[ ]:


test_y = np.reshape(final_y, (total,))


# In[ ]:


#test_y.shape


# In[ ]:


#test_y[:10]


# In[ ]:


# Function to check what proportion split of the data is R regions compared to normal signals

def proportions(y):
    test = list(y)
    points = test.count(1)
    print(points/len(test))
    print((1 - points/len(test)))
    
proportions(test_y)


# In[ ]:


# Create a dense model with a relu activation function that performs binary classification indicating if a point
# Belongs to the R region or not

initializer = "glorot_uniform"
model = Sequential()
model.add(Dense(256, activation ='relu', input_shape = (6,)))
model.add(Dense(1, activation='sigmoid'))
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics=["binary_accuracy", "mae"])
model.summary()


# In[ ]:


# Split set into train and validation randomly -> large enough dataset that simple splitting will work fine
x_train, x_val, y_train, y_val = train_test_split(test_x, test_y, test_size=0.25, shuffle = True)
# Check proportions to ensure they are similar
proportions(y_train)
proportions(y_val)


# In[ ]:


#x_train.shape


# In[ ]:


# Fit the model for 50 epochs using an extremely large batch size due to large input data size
history = model.fit(x_train, y_train, verbose=True, batch_size = 50000, validation_data = (x_val, y_val), epochs = 50)


# In[ ]:


# Get a confusion matrix
preds = model.predict(x_val, verbose = True)

preds = preds.reshape((preds.shape[0],))


# In[ ]:


from sklearn.metrics import confusion_matrix

# Round the probabilities appropriately
preds[preds > 0.5] = 1
preds[preds <= 0.5] = 0

print(confusion_matrix(y_val,preds))


# In[ ]:


#plt.plot(final_x[0][0][:6])


# In[ ]:


# Save the model training history and confusion matrix for future analysis

confusion = confusion_matrix(y_val,preds)
with open(rpeak_dir + 'Non-Weighted/Confusion.pkl'.format(str(i)), 'wb') as f:
    pickle.dump(confusion,f)


# In[ ]:


with open(rpeak_dir + 'Non-Weighted/History.pkl'.format(str(i)), 'wb') as f:
    pickle.dump(history,f)
    
#model.save("RNN R Peak Detection Model_reduce_set.h5")


# In[ ]:


# Repeat the process for a test patient to see what the accuracy is looking like

# Test set is full of patients: 100, 105, 116, 215, 232
# All the other patients can be put into the training set

test = [100]

test_x = np.zeros((len(test), 65,10000,6), dtype=np.float32)
test_y = np.zeros((len(test), 65, 10000), dtype=np.int32)

for o,i in enumerate(test):
    
    print(i)
    
    with open(dir_denoised_data.format(i), 'rb') as f:
        smoothed = pickle.load(f)
    
    with open(fin_dir_peaks.format(i), 'rb') as f:
        peaks = pickle.load(f)
        
    lead_1 = np.zeros((smoothed.shape[0],))
    lead_2 = np.zeros((smoothed.shape[0],))
    
    for c,dual_point in enumerate(smoothed):
        lead_1[c] = dual_point[0]
        lead_2[c] = dual_point[1]
        
    time_stamp = 10000
    
    detrend_lead_1 = np.asarray([lead_1[i: i + time_stamp] for i in range(0, len(lead_1), time_stamp)])
    detrend_lead_2 = np.asarray([lead_2[i: i + time_stamp] for i in range(0, len(lead_2), time_stamp)])
    
    patient_x = np.zeros((65,10000,6), dtype=np.float32)
    
    knot = np.arange(10,10000, 100)
    
    for index,chunk in enumerate(detrend_lead_1):
        
        tck_1 = interpolate.splrep(list(range(time_stamp)), detrend_lead_1[index],t=knot, k = 3, task = -1)
        new_lead_1 = np.array(np.array(detrend_lead_1[index]) - np.array(interpolate.splev(list(range(time_stamp)), tck_1)), dtype = np.float32)
        tck_2 = interpolate.splrep(list(range(time_stamp)), detrend_lead_2[index],t=knot, k = 3, task = -1)
        new_lead_2 = np.array(np.array(detrend_lead_2[index]) - np.array(interpolate.splev(list(range(time_stamp)), tck_2)), dtype = np.float32)
        smoothed = np.array(normalise(new_lead_1),dtype=np.float32)
        smoothed2 = np.array(normalise(new_lead_2),dtype=np.float32)
        deriv = np.array(np.gradient(smoothed), dtype=np.float32)
        deriv = np.array(normalise(deriv),dtype=np.float32)
        deriv2 = np.array(np.gradient(smoothed2),dtype=np.float32)
        deriv2 = np.array(normalise(deriv2),dtype=np.float32)
        deriv_2 = np.array(np.gradient(deriv),dtype=np.float32)
        deriv_2 = np.array(normalise(deriv_2),dtype=np.float32)
        deriv2_2 = np.array(np.gradient(deriv2),dtype=np.float32)
        deriv2_2 = np.array(normalise(deriv2_2),dtype=np.float32)

        x = np.hstack((smoothed,deriv))
        x = np.hstack((x,deriv_2))
        x = np.hstack((x,smoothed2))
        x = np.hstack((x,deriv2))
        x = np.hstack((x,deriv2_2))

        new_x = np.zeros((10000, 6),dtype=np.float32)

        for l in range(smoothed.shape[0]):
            temp = np.array([smoothed[l], deriv[l], deriv_2[l], smoothed2[l], deriv2[l], deriv2_2[l]], dtype=np.float32)
            new_x[l] = temp
            
        patient_x[index] = new_x
        
    y = np.zeros((len(lead_1),))

    buffer = 3
    
    # Set the labels to true in buffer region
    for k in peaks:
        y[k - buffer: k + buffer] = 1
    
    patient_y = np.asarray([y[i: i + time_stamp] for i in range(0, len(y), time_stamp)], dtype=np.int32)
    
    # Now append these to the total
    test_x[o] = patient_x
    test_y[o] = patient_y


# In[ ]:


test_x = np.array(test_x).reshape(650000,6)


# In[ ]:


test_y = np.array(test_y).reshape(650000,)


# In[ ]:


pred = model.predict(test_x, verbose = True)


# In[ ]:


plt.plot(pred[:5000])


# In[ ]:


pred = pred.reshape((pred.shape[0],))


# In[ ]:


def accuracy_metric(new_prob_peaks, peaks):

    sample_window = 2
    counts = 0

    temp_list = list(new_prob_peaks)

    # Loop over the true annotated peaks as well as the predicted peaks
    for true_peak in peaks:
        for predicted_peak in temp_list:

            # If they are within the sample window either side of the true peak then allow it as a count for our case
            if ((predicted_peak >= (true_peak - sample_window)) and (predicted_peak <= (true_peak + sample_window))):
                counts += 1
                # Now remove this peak from the list as it is one-to-one relation so cannot be used again
                temp_list.remove(predicted_peak)
                # Now break from the for loop so we do not do the other components
                break
            else:
                continue
        
    accuracy = (counts/len(new_prob_peaks))
    missing = abs(len(peaks) - round(accuracy*len(new_prob_peaks)))
    wrong = (missing + round((1-accuracy) * len(new_prob_peaks)))
    return wrong


# In[ ]:


prob_peaks, _ = find_peaks(pred, height = 0.05, distance = 10)
new_prob_peaks = prob_peaks[lead_1[prob_peaks] > 0]
accuracy = accuracy_metric(new_prob_peaks, peaks)
#print(accuracy)

#print(len(new_prob_peaks))

#print(len(peaks))

# Plot results
plt.plot(lead_1[:11000])
temp = new_prob_peaks[:41]
plt.plot(temp, lead_1[temp], "x")

