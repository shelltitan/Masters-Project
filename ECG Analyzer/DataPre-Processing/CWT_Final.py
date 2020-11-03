#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This performs a CWT of each beat. As this is going to be a CWT of more widths, there needs to be extra padding
# on the sides of each CWT that is then cut off to get rid of boundary effects from the CWT


# In[1]:


from scipy import signal
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pywt
import PIL
from PIL import Image
from skimage.transform import resize
import math
from pywt._doc_utils import boundary_mode_subplot


# In[2]:
#Directory  variables
dir_array = "D:/arrhythmia-database/RawDataArrays/{}_array.pkl"
dir_segmented_data = 'D:/arrhythmia-database/SegmentedData/{}.pkl'
dir_segments = 'D:/arrhythmia-database/SegmentedData/Segments/{}_segments.pkl'
dir_segment_labels = 'D:/arrhythmia-database/SegmentedData/SegmentLabels/{}_labels.pkl'
dir_segments_CWT = 'D:/arrhythmia-database/SegmentedData/SegmentsCWT/sample_{}'
#%%
##### This is the function that performs the CWT of the beat 

def CWT(beat ,widths=np.arange(1,128)):
    # This performs a CWT of every channel of the segment, then adds and axis and stacks them
    # First find the number of channels in the segment
    num_leads = beat.shape[-1] 
    pad = 350
    # Before performing the CWT, the sides of the array need to be padded:
    padded_1 = np.pad(beat[:,0],(pad,pad),'edge')
    padded_2 = np.pad(beat[:,1],(pad,pad),'edge')
    beat = np.stack((padded_1,padded_2),axis=-1)
    
    # Now loop over every channel and perform the CWT:
    for i in range(num_leads):
        if i == 0:
            cwtmatr, freqs = pywt.cwt(beat[:,i], widths, 'cmor1.5-1')
            cwtha = abs(cwtmatr)
            
            # Need to cut off the ends of the CWT that were added to counter the edge effects
            cwtha = cwtha[:,pad:-pad]
            # Need to resize it to the correct shape
            resized1 = resize(cwtha, (128,128))
            
            # Normalise all the values 
            mean = np.mean(resized1)
            resized1 -= mean
            std = np.std(resized1)
            resized1 /= std
            
            resized1 = np.array(resized1, dtype=np.float32)
            sample = resized1[:,:,np.newaxis]
        else:
            cwtmatr, freqs = pywt.cwt(beat[:,i], widths, 'cmor1.5-1')
            cwtha = abs(cwtmatr)
            cwtha = cwtha[:,pad:-pad]
            resized1 = resize(cwtha, (128,128))
            
            # Normalise all the values 
            mean = np.mean(resized1)
            resized1 -= mean
            std = np.std(resized1)
            resized1 /= std

            resized1 = np.array(resized1, dtype=np.float32)
            resized1 = resized1[:,:,np.newaxis]
            sample = np.concatenate((sample,resized1),axis=-1)
        
    return sample


# In[34]:


with open(dir_array.format(7), 'rb') as f:
        ar = pickle.load(f)
plt.plot(ar[:1000,0])


# In[3]:


with open(dir_segments.format(0), 'rb') as f:
        data = pickle.load(f)


# In[4]:


plt.plot(data[2][:,0])
plt.xlabel('Samples')
plt.ylabel('Millivolt (mV)')
plt.xlabel('Samples')
plt.title('ECG of a Normal Beat')
#plt.savefig('ECG of normal beat',dpi=200)


# In[8]:


CWTed = CWT(data[2])
plt.imshow(CWTed[:,:,0])
#plt.colorbar()
#plt.title('CWT of a Normal Beat')
#plt.savefig('Test Aug',dpi=200)
with open('Test Aug.pkl','wb') as f:
    pickle.dump(CWTed,f)


# In[20]:


np.save('test',CWTed ,allow_pickle = True)


# In[29]:


# loop to find how many samples there will be 
num_samps = 0
for i in range(48):
    with open(dir_segments.format(i), 'rb') as f:
        segments = pickle.load(f)
        
    length = len(segments)
    num_samps += length
print(num_samps)
print('Size of all the data = ', num_samps * 129 , 'kb')


# In[4]:


# This is a loop to perform a CWT of every sample
num_samps_total = 0
sample_labels = []
patient_samples = []
for i in range(48):
    print(i)
    
    with open(dir_segments.format(i), 'rb') as f:
        segments = pickle.load(f)
    with open(dir_segment_labels.format(i), 'rb') as f:
        Labels = pickle.load(f)
    
    
    for j in range(len(segments)):
        # Append the sample label to sample_labels
        sample_labels.append(Labels[j])
        CWTed = CWT(segments[j])
        np.save(dir_segments_CWT.format('num_samps_total'),CWTed ,allow_pickle = True)
        num_samps_total += 1
        
    # Need to know which samples correspond to which patient for later on when we're combining the CNN
    # with the RNN     
    patient_samples.append(num_samps_total)
    # So I know where to start it again if it stops I will print this: 
    print('Number of samples for patient ',i,' is: ',num_samps_total)
with open(dir_segmented_data.format('sample_labels'), 'wb') as f:
        pickle.dump(sample_labels, f)
        
with open(dir_segmented_data.format('patient_samples'), 'wb') as f:
        pickle.dump(patient_samples, f)


# In[1]:


# I need to load in the labels and convert the labels to numbers rather than strings
with open(dir_segmented_data.format('sample_labels'), 'rb') as f:
        Labels = pickle.load(f)
        
#print(Labels)

# Need to turn the labels from strings to numbers that represent different types of beats
label_cat = [0 if b=='N'
            else 1 if b=='L'
            else 2 if b=='R'
            else 3 if b=='A'
            else 4 if b=='a'
            else 5 if b=='J'
            else 6 if b=='S'
            else 7 if b=='V'
            else 8 if b=='F'
            else 9 if b=='!'
            else 10 if b=='e'
            else 11 if b=='j'
            else 12 if b=='E'
            else 13 if b=='/'
            else 14 if b=='f'
            else 15 if b=='x'
            else 16 if b=='Q'
            else 17 for b in Labels]

with open(dir_segmented_data.format('sample_labels_cat'), 'wb') as f:
        pickle.dump(label_cat, f)
        
print(label_cat[:10])


# In[22]:


# Now that the labels are numbers, I can use categorical encoding to make each label a vector of length
# 18 (18 types of labels) with a 1 in the label index.
import keras
from keras.utils.np_utils import to_categorical
one_hot_labels = to_categorical(label_cat)
with open(dir_segmented_data.format('sample_labels_cat'), 'wb') as f:
        pickle.dump(label_cat, f)


# In[23]:


one_hot_labels = to_categorical(label_cat)


# In[30]:


print(len(one_hot_labels))
with open(dir_segmented_data.format('Sample_labels_OHE'), 'wb') as f:
        pickle.dump(one_hot_labels, f)


# In[17]:


# I now need to make a list of all the sample ID's to be read in
Sample_IDs = []
for i in range(len(Labels)):
    Sample_IDs.append('Sample_{}'.format(i))
    
with open(dir_segmented_data.format('Sample_IDs'), 'wb') as f:
        pickle.dump(Sample_IDs, f)


# In[ ]:





# In[ ]:




