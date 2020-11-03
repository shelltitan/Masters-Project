import wfdb
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

db_dir = "D:/mit-bih-arrhythmia-database-1.0.0/"

files = [os.path.splitext(filename)[0] for filename in os.listdir(db_dir) if filename.endswith('.atr')]
files.remove("102-0")
# As they're not in chronological order it's easier to make a list with all the filnames 

#for i,name in enumerate(files):
#    print('For file ',name,' the index is ', i)

#Directory locations
db_dir = "D:/mit-bih-arrhythmia-database-1.0.0/{}" #Raw database directory
#Directory for pickled database files
out_dir_array = "D:/arrhythmia-database/RawDataArrays/{}_array.pkl"
out_dir_label = "D:/arrhythmia-database/RawDataLabels/{}_labels.pkl"
out_dir_label_location = "D:/arrhythmia-database/RawDataLabelLocations/{}_label_locations.pkl"
out_dir_rhythms = "D:/arrhythmia-database/RawDataRhythms/{}_rhythms.pkl"


# This function extracts the ecg signal and converts it into an array
def extract_data(filename):
    record = wfdb.rdrecord(db_dir.format(filename))
    d_signal = record.adc()
    
        
    # This normalises the data so that the max value is 1
    #V_signal = (V_signal - V_signal.min())/(V_signal.max() - V_signal.min())
    return d_signal

for i,name in enumerate(files):
    signal = extract_data(name)
    with open(out_dir_array.format(i), 'wb') as f:
        pickle.dump(signal, f)
        
# This function extracts the labels and their locations (ie the peak locations) for each beat and 
# puts them into a lists
def extract_labels(filename):
    # This reads in the file and converts it to an object called ann
    ann = wfdb.rdann(db_dir.format(filename),'atr',return_label_elements=['symbol'])
    # These two lines return the symbol and the locations 
    labels_symbol = ann.symbol
    locations = ann.sample
    return labels_symbol, locations

for i,name in enumerate(files):
    labels_symbol,peaks = extract_labels(name)
    with open(out_dir_label.format(i), 'wb') as f:
        pickle.dump(labels_symbol, f)
    with open(out_dir_label_location.format(i), 'wb') as f:
        pickle.dump(peaks, f)

# This saves the rhythm annotations 
for i,name in enumerate(files):
    ann = wfdb.rdann(db_dir.format(name),'atr',return_label_elements=['description'])
    rhythm = ann.aux_note
    # This gets rid of the brackets and the '\x00' in the annotations
    for j in range(len(rhythm)):
        rhythm[j] = rhythm[j].rstrip('\x00')
        rhythm[j] = rhythm[j].strip('(')
    with open(out_dir_rhythms.format(i), 'wb') as f:
        pickle.dump(rhythm, f)
