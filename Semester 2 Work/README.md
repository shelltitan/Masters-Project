This Folder contains the work completed in the second semester of the masters project.

Our aim was to automatically detect cardiac arrhythmias from patient raw ECG data. In order to do this we used the MIT-BIH Arrhythmia dataset.

Our method involved combining a two dimensional convolutional neural network with a long short term memory neural network. To our knowledge no literature currently replicates exactly our LSTM method. There have however been similar models set up albeit with different datasets and different rhythm classifications.

In this folder you will find a data pre-processing folder which contains all the files we used to extract, de-noise and de-trend the ECG signal data. As well as an R peak detection folder that we performed late on in the project. Followed by a CNN folder containing the implementation for the CNN. Likewise with the LSTM. Finally there is a testing file which we used to test our two models on our test set.
