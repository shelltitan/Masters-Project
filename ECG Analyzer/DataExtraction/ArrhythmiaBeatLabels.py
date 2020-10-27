# The annotations for the labels include the labels for each beat, the labels for rhythm changes and changes 
# in the signal quality. However, we only need the labels for the individual beats and the position of these
# labels (ie the position of the R peaks) for the CNN and the rhythm labels for the RNN.
# This code gets rid of the other labels and only saves the individual beat labels and get's only the rhythm 
# labels.

import numpy as np
import pickle 
import matplotlib.pyplot as plt
import os

# As they're not in chronological order (and some are missing) it's easier to make a list with all the filnames 
db_dir = "D:/mit-bih-arrhythmia-database-1.0.0/"

files = [os.path.splitext(filename)[0] for filename in os.listdir(db_dir) if filename.endswith('.atr')]
files.remove("102-0")
# As they're not in chronological order it's easier to make a list with all the filnames

# This is a list of the labels that we actually want:

beat_labels = ['N','L','R','A','a','J','S','V','F','!','e','j','E','/','f','x','Q','|']

rhythm_labels = ['AB','AFIB','AFL','B','BII','IVR','N',
                 'NOD','P','PREX','SBR', 'SVTA','T','VFL','VT']

# This bit of code gets the locations and labels for the beats.
with open('D:/arrhythmia-database/RawDataLabels/7_labels.pkl','rb') as f:
    labels = pickle.load(f)
with open('D:/arrhythmia-database/RawDataLabelLocations/7_label_locations.pkl','rb') as f:
    label_location = pickle.load(f)

beat_type = [i for i in labels if i in beat_labels]
beat_index = [i for i,c in enumerate(labels) if c in beat_labels]
peaks = [label_location[i] for i in beat_index]

# This bit of code gets the locations and labels for the rhythms.
with open('D:/arrhythmia-database/RawDataRhythms/7_rhythms.pkl','rb') as f:
    rhythms = pickle.load(f)
with open('D:/arrhythmia-database/RawDataLabelLocations/7_label_locations.pkl','rb') as f:
    label_location = pickle.load(f)
    

rhythm_type = [i for i in rhythms if i in rhythm_labels]
index = [i for i,c in enumerate(rhythms) if c in rhythm_labels]
rhythm_locations = [label_location[i] for i in index]

