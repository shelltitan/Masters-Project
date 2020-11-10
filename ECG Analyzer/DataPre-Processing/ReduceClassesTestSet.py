#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -------- Create Test Set -------- #
# This program adjusts the set aside test set to only select a subset of beats as well as rhythms
# In order to do this, I have to loop through my test set checking labels and inputs to remove ones that
# We are not classifying. These indices need to then be saved in order for Teddy to remove them from his set
# As well as the CWT set.


# In[ ]:


import pickle
import numpy as np

dir_test_data = 'D:/arrhythmia-database/TestData/{}.pkl'

# In[ ]:


# Import the previously set aside test set for our patients
# Test set is full of patients: 100, 105, 116, 215, 232
with open(dir_test_data.format('Rhythm_Locations'), 'rb') as f:
    rhythm_locations = pickle.load(f)
with open(dir_test_data.format('Rhythm_Labels'), 'rb') as f:
    rhythm_labels = pickle.load(f)
with open(dir_test_data.format('Beat_Locations'), 'rb') as f:
    beat_locations = pickle.load(f)
with open(dir_test_data.format('Beat_Labels'), 'rb') as f:
    beat_labels = pickle.load(f)


# In[ ]:


#print(len(beat_labels))


# In[ ]:


# Need to loop over all the rhythm locations and remove the half point 325000.0
for i in range(len(rhythm_locations)):
    if(rhythm_locations[i][-1] == 325000.0):
        del rhythm_locations[i][-1]


# In[ ]:


#i = 7
#print(rhythm_labels[i])
#print(rhythm_locations[i])
#print(beat_locations[i][:10])

# In order to get the locations and labels in the correct form, need to remove certain elements
# And adapt the set

del rhythm_locations[3][0]
del rhythm_labels[3][0]
del rhythm_locations[6][0]
del rhythm_labels[6][0]
del rhythm_locations[7][0]
rhythm_locations[4][0] = 322000


# In[ ]:


# Swap patients around to get in the order 4,14,24,28....
# swap the 14 -> 4
rhythm_locations[2], rhythm_locations[0] = rhythm_locations[0], rhythm_locations[2]
rhythm_labels[2], rhythm_labels[0] = rhythm_labels[0], rhythm_labels[2]
beat_locations[2], beat_locations[0] = beat_locations[0], beat_locations[2]
beat_labels[2], beat_labels[0] = beat_labels[0], beat_labels[2]
# swap the 24 -> 14
rhythm_locations[2], rhythm_locations[1] = rhythm_locations[1], rhythm_locations[2]
rhythm_labels[2], rhythm_labels[1] = rhythm_labels[1], rhythm_labels[2]
beat_locations[2], beat_locations[1] = beat_locations[1], beat_locations[2]
beat_labels[2], beat_labels[1] = beat_labels[1], beat_labels[2]


# In[ ]:


# Now pre-process the data completely
# Delete last beat as this might have issues with ECG disconnection, not a complete beat etc
for c,i in enumerate(beat_locations):
    if(c == 0):
        continue
    else:
        del i[-1]
for c,j in enumerate(beat_labels):   
    if(c == 0):
        continue
    else:
        del j[-1]

# -------- Split Into Samples -------- #
# Choose a sample size - The number of beat labels you are going to send in
sample_size = 10

# Now we need to create sublists for each patient with 25 beat labels in
full_list = []
for j in beat_labels:
    samples = [j[i * sample_size:(i + 1) * sample_size] for i in range((len(j) + sample_size - 1) // sample_size )]
    full_list.append(samples)
    
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
            
# Now we need to create sublists for each patient with 10 beat label locations in
location_list = []
for j in beat_locations:
    samples = [j[i * sample_size:(i + 1) * sample_size] for i in range((len(j) + sample_size - 1) // sample_size )]
    location_list.append(samples)
    
sample_labels = []

# First pick a patient
for i,patient in enumerate(rhythm_locations):
    
    #print(i)
    
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
                    #print(len(rhythm_labels[i]))
                    #print(len(rhythm_locations[i]))
                   # print(rhythm_labels[i][c])
                    rhythm_after.append(rhythm_labels[i][c])
                    
        #print(rhythm_after)
                
        patient_beat_labels.append(rhythm_after)
        #print(patient_beat_labels[-1])
       # print(beat_location)
        #print(i)
        
    sample_labels.append(patient_beat_labels)

    
# Likewise split the labels into windows of 10 labels at a time
full_list_labels = []
for j in sample_labels:
    samples_labels = [j[i * sample_size:(i + 1) * sample_size] for i in range((len(j) + sample_size - 1) // sample_size )]
    full_list_labels.append(samples_labels)
   # print(samples_labels)
    
    
# Data saves the labels with brackets we just need the letter so we remove the bracket notation eg
# (N -> N
x = []
# Per patient
for c,i in enumerate(full_list_labels):
   # print(c)
    patient = []
    # Each sample
    for j in i:
       # print(j)
        new = []
        # Get rid of annoying list notation
        for k in j:
            print(k[0])
            
            # Check for errors
            try:
                new.append(k[0])
            except IndexError:
                print(k)
            #new.append(k[0])
        patient.append(new)
    x.append(patient)
    
# Function to one hot encode the rhythm labels
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

# Change all the string labels into integer labels instead
complete_labels = []
for i in x:
    patient_l = []
    for j in i:
        patient_l.append(create_complete_label(j))
        
    complete_labels.append(patient_l)
    
y = []

# Loop over every sample and create a multilabel vector for it and append it to a list
for c,patient in enumerate(complete_labels):
    
    for sample in patient:
        
        # Create a label vector
        temp = np.zeros((15,))
        
        # Find which numbers are present and hence which rhythms are present, then these indices need to be set to 1
        for item in sample:
            temp[item] = 1
            
        y.append(temp)
        
inputs = []
for i in integer_samples:
    for j in i:
        inputs.append(j)


# In[ ]:


window_beat_labels = []
window_beat_locations = []

# Split up the labels and locations into windows of 10
for j in range(8):

    new_beat_labels = [beat_labels[j][i:i+10] for i in range(0, len(beat_labels[j]), 10)]
    #print(new_beat_labels[0])
    new_beat_locations = [beat_locations[j][i:i+10] for i in range(0, len(beat_locations[j]), 10)]
    #print(new_beat_locations[0])
    #print(len(new_beat_labels))
    #print(len(new_beat_locations))
    window_beat_labels += new_beat_labels
    window_beat_locations += new_beat_locations


# In[ ]:


window_beat_locations = np.array(window_beat_locations)
#print(window_beat_locations.shape)
#print(inputs.shape)

# Check as some of them will not be padded
for i in window_beat_locations:
    if(len(i) < 10):
        print(i)


# In[ ]:


indices_removed_1 = []


# In[ ]:


# Now we need to remove any samples from the test data that has a peak we cannot use
# Peaks we can use: N L R A V ! j / f x

can_use = [0,1,2,3,7,9,11,13,14,15]

total_x = []
new_y = []
new_inputs = []
total_beat_locations = []
new_beat_locations = []

index = -1
# Loop through inputs and take out the beats with 99 in first (Ie do not use the very final window instead of padding it)
for c,i in enumerate(inputs):
    present = False
    if(i.count(99) >0):
        present = True
    if(present == False):
        for number in i:
            index+=1
        new_inputs.append(i)
        new_y.append(y[c])
        new_beat_locations.append(window_beat_locations[c])
    else:
        for number in i:
            if(number!= 99):
                index+=1
                indices_removed_1.append(index)


# In[ ]:


# Now loop over and take out all instances of beats we cannot have and save the indices
index = -1
total_y = []
indices_removed_2 = []
for c,i in enumerate(new_inputs):
    present = False
    for number in i:
        if(number not in can_use):
            present = True
            break
            
    if(present == False):
        total_x.append(new_inputs[c])
        total_y.append(new_y[c])
        total_beat_locations.append(new_beat_locations[c])
        for number in i:
            index+= 1
    else:
        for number in i:
            index += 1
            indices_removed_2.append(index)


# In[ ]:


# Now loop over and remove beats that are in rhythms we cannot predict

no_use = [0,2,5,7,9,12,13,14]

final_x = []
final_y = []
final_beat_locations = []
windows_removed = []

for c,i in enumerate(total_y):
    # If none of the incorrect rhythm indices are filled then we can use these samples
    if ((i[no_use] == 1).any() == False):
        final_x.append(total_x[c])
        final_y.append(total_y[c])
        final_beat_locations.append(total_beat_locations[c])
    # Else save the index of that window to remove
    else:
        indices = list(np.arange((c*10), ((c*10)+10)))
        windows_removed += (indices)


# In[ ]:


# Save these indices and send them over to Teddy who can remove them from his CNN test set

with open('first_indices_removed.pkl', 'wb') as f:
    pickle.dump(indices_removed_1,f)
with open('second_indices_remove.pkl', 'wb') as f:
    pickle.dump(indices_removed_2,f)
with open('third_indices_remove.pkl', 'wb') as f:
    pickle.dump(windows_removed,f)


# In[ ]:


#print(final_x[0])
#print(final_y[0])
#print(final_beat_locations[0])


# In[ ]:


# Need to change my x column now to only include new numbers corresponding to the beats the CNN
# Can predict as it is now fitting a 10 vector of one hot encoded so:
# beats 0,1,2,3,7,9,11,13,14,15 become
# beats 0,1,2,3,4,5,6,7,8,9 elements of the vector

final_x = np.array(final_x)
#print(final_x.shape)
final_x[final_x == 13] = 7
final_x[final_x == 7] = 4
final_x[final_x == 9] = 5
final_x[final_x == 11] = 6
final_x[final_x == 14] = 8
final_x[final_x == 15] = 9


# In[ ]:


#print(final_x[:2])


# In[ ]:


final_beat_locations = np.array(final_beat_locations)
final_y = np.array(final_y)
#print(final_x.shape)
#print(final_y.shape)
#print(final_beat_locations.shape)


# In[ ]:


# Save this new modified test set ready to use
with open('RNN Models/RNN_Test_x.pkl', 'wb') as f:
    pickle.dump(final_x,f)
with open('RNN Models/RNN_Test_y.pkl', 'wb') as f:
    pickle.dump(final_y,f)
with open('RNN Models/RNN_Test_Beat_Locations.pkl', 'wb') as f:
    pickle.dump(final_beat_locations,f)


# In[ ]:


# This is just a check for me and my partner to ensure we definitely have the same set and there is no errors
# Gives us the proportions of the beats in the sample

count_1,count_2,count_3,count_4,count_5,count_6,count_7,count_8,count_9,count_10,count_11,count_12,count_13,count_14,count_15,count_16,count_17,count_18 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

options = {0 : count_1,
           1 : count_2,
           2 : count_3,
           3 : count_4,
           4 : count_5,
           5 : count_6,
           6 : count_7,
           7 : count_8,
           8 : count_9,
           9 : count_10,
          10 : count_11,
          11 : count_12,
          12 : count_13,
          13 : count_14,
          14 : count_15,
          15 : count_16,
          16 : count_17,
          17 : count_18
}

for i in final_x:
    for number in i:
        if(number == 99):
            continue
        else:
            options[number] += 1


# In[ ]:


#print(options)


# In[ ]:


#print(beat_locations[0])


# In[ ]:


# This is a check to see the proportions of the amount of rhythms within the sample set

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


new = proportions(final_y)
print(new)


# In[ ]:


# Take out multiclass and see how proportions changes, which rhythms are removed
multiclass_y = []
for c,i in enumerate(final_y):
    
    if(list(i).count(1) == 1):
        multiclass_y.append(i)
        
newer = proportions(multiclass_y)
print(newer)

