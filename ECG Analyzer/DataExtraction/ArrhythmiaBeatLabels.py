# The annotations for the labels include the labels for each beat, the labels for rhythm changes and changes 
# in the signal quality. However, we only need the labels for the individual beats and the position of these
# labels (ie the position of the R peaks) for the CNN and the rhythm labels for the RNN.
# This code gets rid of the other labels and only saves the individual beat labels and get's only the rhythm 
# labels.

import numpy as np
import pickle 
import matplotlib.pyplot as plt
import os

db_dir = "D:/mit-bih-arrhythmia-database-1.0.0/" #Data base directory
# Directory locations
dir_label = "D:/arrhythmia-database/RawDataLabels/{}_labels.pkl"
dir_label_locations = "D:/arrhythmia-database/RawDataLabelLocations/{}_label_locations.pkl"
dir_label_rhythms = 'D:/arrhythmia-database/RawDataRhythms/{}_rhythms.pkl'
#Final directory
fin_dir_peaks = 'D:/arrhythmia-database/RawDataFinal/Peaks/{}_peaks.pkl'
fin_dir_beat_labels = 'D:/arrhythmia-database/RawDataFinal/BeatLabels/{}_beat_labels.pkl'
fin_dir_rhythm_labels = 'D:/arrhythmia-database/RawDataFinal/RhythmLabels/{}_rhythm_labels.pkl'
fin_dir_rhythm_locations = 'D:/arrhythmia-database/RawDataFinal/RhythmLocations/{}_rhythm_locations.pkl'

# Make list of all filenames used 
files = [os.path.splitext(filename)[0] for filename in os.listdir(db_dir) if filename.endswith('.atr')]
files.remove("102-0")


# This is a list of the labels that we actually want:

beat_labels = ['N','L','R','A','a','J','S','V','F','!','e','j','E','/','f','x','Q','|']

rhythm_labels = ['AB','AFIB','AFL','B','BII','IVR','N',
                 'NOD','P','PREX','SBR', 'SVTA','T','VFL','VT']

# This bit of code gets the locations and labels for the beats.
with open(dir_label.format(str(7)),'rb') as f:
    labels = pickle.load(f)
with open(dir_label_locations.format(str(7)),'rb') as f:
    label_location = pickle.load(f)

beat_type = [i for i in labels if i in beat_labels]
beat_index = [i for i,c in enumerate(labels) if c in beat_labels]
peaks = [label_location[i] for i in beat_index]

# This bit of code gets the locations and labels for the rhythms.
with open(dir_label_rhythms.format(str(7)),'rb') as f:
    rhythms = pickle.load(f)
with open(dir_label_locations.format(str(7)),'rb') as f:
    label_location = pickle.load(f)
    

rhythm_type = [i for i in rhythms if i in rhythm_labels]
index = [i for i,c in enumerate(rhythms) if c in rhythm_labels]
rhythm_locations = [label_location[i] for i in index]


# This is a loop that does the previous code for every file and saves the arrays
for i,name in enumerate(files):
    with open(dir_label.format(i),'rb') as f:
        labels = pickle.load(f)
    with open(dir_label_locations.format(i),'rb') as f:
        label_location = pickle.load(f)

    beat_type = [j for j in labels if j in beat_labels]
    beat_index = [j for j,c in enumerate(labels) if c in beat_labels]
    peaks = [label_location[j] for j in beat_index]
    
    
    with open(dir_label_rhythms.format(i),'rb') as f:
        rhythms = pickle.load(f)


    rhythm_type = [j for j in rhythms if j in rhythm_labels]
    index = [j for j,c in enumerate(rhythms) if c in rhythm_labels]
    rhythm_locations = [label_location[j] for j in index]
    
    
    with open(fin_dir_peaks.format(i), 'wb') as f:
        pickle.dump(peaks, f)
        
    with open(fin_dir_beat_labels.format(i), 'wb') as f:
        pickle.dump(beat_type, f)
        
    with open(fin_dir_rhythm_labels.format(i), 'wb') as f:
        pickle.dump(rhythm_type, f)
        
    with open(fin_dir_rhythm_locations.format(i), 'wb') as f:
        pickle.dump(rhythm_locations, f)
        
with open(fin_dir_rhythm_locations.format('2'),'rb') as f:
        rh = pickle.load(f)
print(rh)