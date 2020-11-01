#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -------- Code Outline -------- #
# Code splits the de-noised data into segments
# Each segment will contain only one heart beat
# As result of CA's, these will be beats of different lengths
# Depending upon the heart rate at that time
# In order to do this we will implement the beats in such a way that they begin 2/3rd
# Of the previous beat and end at 1/3rd of the next beat

import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os

# In[41]:

db_dir = "D:/mit-bih-arrhythmia-database-1.0.0/"

files = [os.path.splitext(filename)[0] for filename in os.listdir(db_dir) if filename.endswith('.atr')]
files.remove("102-0")

#Directory locations
fin_dir_peaks = 'D:/arrhythmia-database/RawDataFinal/Peaks/{}_peaks.pkl'
fin_dir_beat_labels = 'D:/arrhythmia-database/RawDataFinal/BeatLabels/{}_beat_labels.pkl'
dir_denoised_data = 'D:/arrhythmia-database/DenoisedData/{}_de-noised.pkl'
dir_segments = 'D:/arrhythmia-database/SegmentedData/Segments/{}_segments.pkl'
dir_segment_labels = 'D:/arrhythmia-database/SegmentedData/SegmentLabels/{}_labels.pkl'
#%%
for i,name in enumerate(files):
    # Load in the peaks
    with open(fin_dir_peaks.format(i), 'rb') as f:
        peaks = pickle.load(f)

    # Load in the de-noised original data
    with open(dir_denoised_data.format(i), 'rb') as f:
        Data = pickle.load(f)
       
    # -------- Segmentation -------- #
    # Split the beats up into segments based on the peak array

    # Find the 2/3rd point of the first peak distance
    # All starting from the 2/3rd point of the previous interval
    # and ending at 1/3 of the next interval 

    # 2/3rd into first peak distance
    initial_point = int(round(peaks[0] * (2/3)))

    # Split the data into segmented chunks
    segments = []
    # Loop over all the beats, splitting into segments
    # Do this until every peak has been covered (every beat)

    beats = len(peaks)
    for j in range(1,beats):
        #print(initial_point)
        end_point = peaks[j - 1] + int(round(peaks[j] - peaks[j - 1]) * (2/3))
        segments.append(Data[initial_point:end_point,:])

        # Now reset the initial point as 2/3rd of the next RR interval
        initial_point = (peaks[j - 1] + int(round((peaks[j] - peaks[j - 1]) * (2/3))))
        
    with open(dir_segments.format(i), 'wb') as f:
        pickle.dump(segments, f)
       
    
    
    # I also need to delete the last label in the labels list:
    with open(fin_dir_beat_labels.format(i), 'rb') as f:
        Labels = pickle.load(f)
        
    del Labels[-1]
    
    with open(dir_segment_labels.format(i), 'wb') as f:
        pickle.dump(Labels, f)
  
# In[42]:


with open(dir_denoised_data.format(1), 'rb') as f:
        Data = pickle.load(f)
        
plt.plot(Data[600:1200])


# In[43]:


with open(dir_segments.format(1), 'rb') as f:
        seg = pickle.load(f)
        plt.plot(seg[2])


# In[59]:


print(len(segments))
print(len(Labels))


# In[18]:


plt.plot(Data[300:750])


# In[20]:


plt.plot(segments[2])


# In[64]:


with open(dir_segments.format(0), 'rb') as f:
        segments = pickle.load(f)
segments = np.array(segments)
print(segments[0].shape[-1])


# In[61]:


# Because all the segments are different lengths, when indexing it you have to think about it as a list of arrays.
# So you use list indexing style to pick which ever segment, then use the numpy indexing style to get the lead.
plt.plot(segments[0][:,0])


# In[1]:


with open(dir_segments.format(0), 'rb') as f:
        segments = pickle.load(f)


# In[ ]:




