#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -------- code outline ------- #
# This code starts with a function to remove the noise from the signal. Then there's a loop that 
# reads in each file, performs the noise removal, then saves the noise removed signal as a pickle file


# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import pickle
import pywt 
import scipy
from scipy import signal
import os

#%%
#Directory locations of data
dir_array = "D:/arrhythmia-database/RawDataArrays/{}_array.pkl"
dir_denoised_data = 'D:/arrhythmia-database/DenoisedData/{}_de-noised.pkl'
db_dir = 'D:/mit-bih-arrhythmia-database-1.0.0/'

# In[66]:


#  I need to de-noise both lead of the ecg signal then put it back into one array


# -------- Noise Filtering -------- #
# This is the function that performs the noise removal of the signal
def DWT(signal, thresh = 0.003, wavelet="db4",num_leads = 2):
    # Make an empty array for the final de noised signal
    de_noised = np.zeros_like(signal)
    
    # Loop over the number of leads, filter each one then put into de_noised array
    for j in range(num_leads):
        # This performs the de-noising
        threshold = thresh*np.nanmax(signal[:,j])
        coeff = pywt.wavedec(signal[:,j], wavelet, mode="smooth")
        coeff[14:] = (pywt.threshold(i, value=threshold, mode="soft" ) for i in coeff[14:])
        reconstructed_signal = pywt.waverec(coeff, wavelet, mode="smooth" )
        
        de_noised[:,j] = reconstructed_signal
    
    
    return de_noised


# In[67]:


# Need to loop over every patient and put the de-noised data into the de-noised folder
# Easiest by looping over the list of files 


files = [os.path.splitext(filename)[0] for filename in os.listdir(db_dir) if filename.endswith('.atr')]
files.remove("102-0")


for i,name in enumerate(files):
    with open(dir_array.format(i),'rb') as f:
        data = pickle.load(f)
        
    denoised = DWT(data,thresh=0.2,wavelet="db4",num_leads=2)
    
    with open(dir_denoised_data.format(i), 'wb') as f:
        pickle.dump(denoised, f)
    


# In[62]:


with open(dir_array.format(2),'rb') as f:
    test = pickle.load(f)


# In[63]:


new_data = DWT(test, thresh = 0.2, wavelet="db4",num_leads = 2)


# In[65]:


get_ipython().run_line_magic('matplotlib', 'qt')
plt.figure(1)
plt.plot(new_data[:1000,0])
plt.figure(2)
plt.plot(test[0:1000,0])
# plt.plot(test[:200,0])


# In[ ]:





# In[ ]:





# In[4]:


plt.figure(3)
plt.plot(test[:250,0])


# In[67]:


def DWT2(signal, thresh = 0.003, wavelet="db4",num_leads = 2):
    # Make an empty array for the final de noised signal
    de_noised = np.zeros_like(signal)
    
    # Loop over the number of leads, filter each one then put into de_noised array
    for j in range(num_leads):
        # This performs the de-noising
        coeff = pywt.wavedec(signal[:,j], wavelet, mode="smooth")
        threshold = 0
        
        coeff[15:] = (pywt.threshold(i, value=threshold, mode="less" ) for i in coeff[15:])
        reconstructed_signal = pywt.waverec(coeff, wavelet, mode="smooth" )
        
        de_noised[:,j] = reconstructed_signal
    
    
    return de_noised


# In[68]:


new = DWT2(test)


# In[69]:


plt.figure(4)
plt.plot(new[:250,0])


# In[ ]:




