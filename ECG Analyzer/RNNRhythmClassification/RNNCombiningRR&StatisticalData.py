#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -------- Code Outline -------- #
# Create a model that implements an RNN using output predictions from a CNN in order to identify rhythm changes.
# In this instance we have the choice to combine either RR intervals or RR intervals as well as statistical data


# In[ ]:


import numpy as np
import pickle
import keras
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix, hamming_loss, jaccard_score
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from skmultilearn.model_selection import IterativeStratification

#%%
#Directory varbiables
dir_denoised_data = 'D:/arrhythmia-database/DenoisedData/{}_de-noised.pkl'
fin_dir_peaks = 'D:/arrhythmia-database/RawDataFinal/Peaks/{}_peaks.pkl'
fin_dir_beat_labels = 'D:/arrhythmia-database/RawDataFinal/BeatLabels/{}_beat_labels.pkl'
fin_dir_rhythm_labels = 'D:/arrhythmia-database/RawDataFinal/RhythmLabels/{}_rhythm_labels.pkl'
fin_dir_rhythm_locations = 'D:/arrhythmia-database/RawDataFinal/RhythmLocations/{}_rhythm_locations.pkl'
dir_test_data = 'D:/arrhythmia-database/TestData/{}.pkl'
dir_RNN_Stats_val = 'D:/arrhythmia-database/RNN_RR_Stats_10_Fold_Validation'

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
    #print(index)


# In[ ]:


# Function that finds the closest rhythm location to the beat location

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


#print(len(rhythm_locations))
#print(len(test_rhythm_locations))
#print(test_rhythm_labels[2])
#print(test_rhythm_locations[2])
#print(rhythm_locations[-1])
#print(rhythm_labels[-1])


# In[ ]:


# Save this data
with open(dir_test_data.format('Rhythm_Locations'), 'wb') as f:
    pickle.dump(test_rhythm_locations,f)
with open(dir_test_data.format('Rhythm_Labels'), 'wb') as f:
    pickle.dump(test_rhythm_labels,f)
with open(dir_test_data.format('Beat_Locations'), 'wb') as f:
    pickle.dump(test_beat_locations,f)
with open(dir_test_data.format('Beat_Labels'), 'wb') as f:
    pickle.dump(test_beat_labels,f)


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
#print(integer_samples[0][1])


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

# Now we need to create sublists for each patient with 25 beat label locations in
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


import hrvanalysis
from hrvanalysis.extract_features import get_time_domain_features,get_frequency_domain_features,get_geometrical_features, get_poincare_plot_features, get_sampen
from scipy.stats import entropy

RR_distances = np.array(RR_distances)

RR_distances = RR_distances.reshape((RR_distances.shape[0], sample_size))

#RR_distances = RR_distances.reshape(RR_distances.shape[0], sample_size) ORIGINAL

# Find different statistical information about the RR time sequence evolution. These are in the form of time domains,
# Frequency Domains as well as non-linear features

indexes_to_remove = []
mean_nni = []
sdnn = []
sdsd = []
pnni_20 = []
pnni_50 = []
range_nni = []
total_power = []
lf = []
hf = []
lf_hf_ratio = []
sd1 = []
sample_entropy = []

# Calculate it for each window in the RR distances matrix

for c,segment in enumerate(RR_distances):
    
    #print(c/len(RR_distances) * 100)
    
    # Make a copy so we do not adjust the RR distances matrix accidentally
    temp = segment.copy()
    # Only interested in the values which are not 0 (ie the paddings)
    temp = list(filter(lambda a: a != 0, temp))
    
    # If we can only get one RR interval then we are at the last few peaks in the patients ECG signal, We should save this indices
    # And just remove them from the sets, the sample entropy requires minimum 3 RR intervals so remove any less than that
    if(len(temp) <= 3):
        indexes_to_remove.append(c)
        continue
    
    # Now loop over the temporary segment and find the time series evolution of the means
    
    time_series_mean_nni = []
    time_series_sdnn = []
    time_series_sdsd = []
    time_series_pnni_20 = []
    time_series_pnni_50 = []
    time_series_range_nni = []
    time_series_total_power = []
    time_series_lf = []
    time_series_hf = []
    time_series_lf_hf_ratio = []
    time_series_sd1 = []
    time_series_sample_entropy = []
    
    for i in range(sample_size):
        
        if(i == 0):
            
            time_series_mean_nni.append(0)
            time_series_sdnn.append(0)
            time_series_sdsd.append(0)
            time_series_pnni_20.append(0)
            time_series_pnni_50.append(0)
            time_series_range_nni.append(0)
            time_series_total_power.append(0)
            time_series_lf.append(0)
            time_series_hf.append(0)
            time_series_lf_hf_ratio.append(0)
            time_series_sd1.append(0)
            time_series_sample_entropy.append(0)
            
        else:
            
            temp_RR = temp[:(i+1)]
            
            #print(temp_RR)
            
            # Now find the features and extract the ones we require
            # For time domain we will use Mean,SDNN,RMSSD,SDSD, pNN20, pNN50
            # Frequency domain we will use totalPSD, HF
            # Non linear features we will use SD1/SD2 and sample entropy

            time_domain = get_time_domain_features(temp_RR)

            # Time domain components
            time_series_mean_nni.append(time_domain['mean_nni'])
            time_series_sdnn.append(time_domain['sdnn'])
            time_series_sdsd.append(time_domain['sdsd'])
            time_series_pnni_20.append(time_domain['pnni_20'])
            time_series_pnni_50.append(time_domain['pnni_50'])
            time_series_range_nni.append(time_domain['range_nni'])

            frequency_domain = get_frequency_domain_features(temp_RR, 'welch', 360)

            # Frequency domain components
            time_series_total_power.append(frequency_domain['total_power'])
            time_series_hf.append(frequency_domain['hf'])
            
            # Non linear features require minimum 3 RR intervals
            
            if (i  == 1):
                
                time_series_sd1.append(0)
                time_series_sample_entropy.append(0)
                
            else:

                poincare = get_poincare_plot_features(temp_RR)

                # Get non linear features
                time_series_sd1.append(poincare['sd1'])

                RR_entropy = entropy(temp_RR)

                time_series_sample_entropy.append(RR_entropy)
        
    mean_nni.append(time_series_mean_nni)
    sdnn.append(time_series_sdnn)
    sdsd.append(time_series_sdsd)
    pnni_20.append(time_series_pnni_20)
    pnni_50.append(time_series_pnni_50)
    range_nni.append(time_series_range_nni)
    total_power.append(time_series_total_power)
    hf.append(time_series_hf)
    sd1.append(time_series_sd1)
    sample_entropy.append(time_series_sample_entropy)


# In[ ]:


# Now fit these all together and normalise
from sklearn.preprocessing import normalize
mean_nni = np.array(mean_nni)
mean_nni = normalize(mean_nni)
sdnn = np.array(sdnn)
sdnn = normalize(sdnn)
sdsd = np.array(sdsd)
sdsd = normalize(sdsd)
pnni_20 = np.array(pnni_20)
pnni_20 = normalize(pnni_20)
pnni_50 = np.array(pnni_50)
pnni_50 = normalize(pnni_50)
range_nni = np.array(range_nni)
range_nni = normalize(range_nni)
total_power = np.array(total_power)
total_power = normalize(total_power)
hf = np.array(hf)
hf = normalize(hf)
sd1 = np.array(sd1)
sd1 = normalize(sd1)
sample_entropy = np.array(sample_entropy)
sample_entropy = normalize(sample_entropy)

#print(sample_entropy.shape)
#print(mean_nni[0])
#print(sdnn[0])


# In[ ]:


#print(RR_distances[0])
#print(x[0])
#print(SVM_RR_distances[0])
#print(svm_x[0])

# Remove the windows which we could not use as they were less than 3 in length
svm_x = np.delete(x, indexes_to_remove)
print(indexes_to_remove)
SVM_RR_distances = np.delete(RR_distances, indexes_to_remove)
svm_y = np.delete(y, indexes_to_remove)

# Reshape them
svm_x = svm_x.reshape((svm_x.shape[0], svm_x.shape[1]))
SVM_RR_distances = SVM_RR_distances.reshape((SVM_RR_distances.shape[0],SVM_RR_distances.shape[1]))

# Normalise the RR intervals
SVM_RR_distances = normalize(SVM_RR_distances)

#print(svm_x[0])
#print(svm_x.shape)
#print(SVM_RR_distances.shape)
#print(SVM_RR_distances[0])


# In[ ]:


# Now stack on the statistics features
total_x = np.stack([svm_x,SVM_RR_distances,mean_nni, sdnn, sdsd, pnni_20, pnni_50, range_nni, total_power, hf, sd1, sample_entropy], axis = 2)
#print(total_x.shape)
#print(total_x[0])
#print(svm_y.shape)
#print(svm_y[0])


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


data_split = proportions(svm_y)


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


# As some labels are not present frequently in the sample, we need to implement a k fold validation scheme
# In the hope that this will produce a more representative set.

import math

batch_size = 28

# Many thanks to creators of iterative stratification and scikit-multilearn
# Reference :
# If you use this method to stratify data please cite both:
# 1 -> Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. Machine Learning and Knowledge Discovery in Databases, 145-158. http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
# 2 -> Piotr Szymański, Tomasz Kajdanowicz ; Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017. http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html
# Found on http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html


from skmultilearn.model_selection import IterativeStratification
 
# The more splits you do the more likely the data is representative however this comes with trade off that
# You will have a lower percentage of data in your test set as well as the fact that it will take significantly
# Longer computationally. Thankfully the RNN is very fast as we are just using sequences so a 10-fold validation
# Will be sufficient and leaves 90% of the data for training purposes

n_split = 10
 
k_fold = IterativeStratification(n_splits = n_split, order=1)

i = 1

# This is just to check that splits are somewhat correct and then crucially saves the data set splits for each fold
# Into a file for analysis later

# Save the training and validation indices for each fold so that we can use them on the server
train_indices = []
val_indices = []

for train, val in k_fold.split(total_x, svm_y):
    print("Fold" + str(i))
    temp_1 = proportions(y[train])
    temp_2 = proportions(y[val])
    print("Train set")
    print(temp_1)
    print("Validation set")
    print(temp_2)
    i += 1
    train_indices.append(train)
    val_indices.append(val)


# In[ ]:


# Now save the indices for GPU use on server
#with open('RNN Training Indices.pkl', 'wb') as f:
 #   pickle.dump(train_indices, f)
#with open('RNN Validation Indices.pkl', 'wb') as f:
 #   pickle.dump(val_indices, f)


# In[ ]:


# Implement a weighted loss that will account for the multilabel data being imbalanced
def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights


# In[ ]:


# Implement a loss function that accounts for these weights for multilabel data

import keras.backend as K
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


# In[ ]:


# Data folds are ready to go so now we need to train on every fold and save a couple of things:
# 1. Data set for each fold
# 2. Actual data for each fold
# 3. Accuracies on validation and training set for each fold
# 4. Average accuracy and loss over 10 folds

i = 1

# Need to save the scores for each fold evaluation
scores = np.zeros((n_split, 4))

from keras import optimizers

optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

for train, val in k_fold.split(total_x, svm_y):
    
    print("Fold " + str(i))
    
    train_x = total_x[train]
    train_y = svm_y[train]
    val_x = total_x[val]
    val_y = svm_y[val]
    
    #class_weights = calculating_class_weights(train_y)
    
    # Find new proportions and save them
    total = []
    data_split_train = proportions(train_y)
    total.append(data_split_train)
    data_split_val = proportions(val_y)
    total.append(data_split_val)
    
    with open(dir_RNN_Stats_val + '/Data_Splits_For_Fold_{}_Weighted.pkl'.format(str(i)), 'wb') as f:
        pickle.dump(total,f)
    
    #train_x,train_y = fit_to_batch(train_x,train_y,batch_size)
    #val_x, val_y = fit_to_batch(val_x,val_y,batch_size)
    
    # Create the model
    model = Sequential()
    model.add(Bidirectional(LSTM(units = 256,
                            stateful = False,
                            recurrent_dropout = 0.2,
                            activation = 'sigmoid'),
                            input_shape = (sample_size,total_x.shape[-1])))
    
    
    model.add(Dense(15, activation = 'sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Fit to data
    
    history = model.fit(train_x, train_y, verbose = 1, batch_size = batch_size, validation_data = (val_x, val_y), epochs = 50, shuffle = True)
    
    # Save history for each fold
    with open(dir_RNN_Stats_val + '/History_For_Fold_{}_Weighted.pkl'.format(str(i)), 'wb') as f:
        pickle.dump(history,f)
    
    # Save the evaluate values for later
    print("Evaluation on Training set")
    
    train_scores = model.evaluate(train_x, train_y, batch_size = batch_size)
    scores[i - 1][0] = train_scores[0]
    scores[i - 1][1] = train_scores[1]
    #print(train_scores)
    
    print("Evaluation on Validation set")
    
    val_scores = model.evaluate(val_x, val_y, batch_size = batch_size)
    scores[i - 1][2] = val_scores[0]
    scores[i - 1][3] = val_scores[1]
    predictions = model.predict(val_x)
    # Turn to prediction multilabel outputs
    predictions[predictions>=0.5] = 1
    predictions[predictions<0.5] = 0
    #print(predictions[0])
    # Add in hamming loss, jaccard score, recall, precision, f1 score, then save multilabel confusion matrix
    scikit_scores = []
    scikit_scores.append(hamming_loss(val_y, predictions))
    scikit_scores.append(jaccard_score(val_y, predictions, average = None))
    scikit_scores.append(recall_score(val_y, predictions, average = None))
    scikit_scores.append(precision_score(val_y, predictions, average = None))
    scikit_scores.append(f1_score(val_y, predictions, average = None))
    confusion = multilabel_confusion_matrix(val_y, predictions)
    print(confusion)
    print(f1_score(val_y, predictions, average = None))
    #print(val_scores)
    #print(scikit_scores)
    #print(confusion)
    
    # Save the scikit parameters for this fold
    with open(dir_RNN_Stats_val + '/Scikit_Scores_For_Fold_{}_Weighted.pkl'.format(str(i)), 'wb') as f:
        pickle.dump(scikit_scores,f)
    
    # Save the confusion matrix for this fold
    with open(dir_RNN_Stats_val + '/Confusion_Matrix_For_Fold_{}_Weighted.pkl'.format(str(i)), 'wb') as f:
        pickle.dump(confusion,f)
    
    i += 1
    
np.save(dir_RNN_Stats_val + '/Scores_Weighted.npy', scores, allow_pickle = True)

