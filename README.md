# Masters-Project
Automatic Detection of Cardiac Arrhythmias
For Henggui:

The following list is in order of completion as well as brief descriptions.

The Poincare plot codes were done at the very start of the project alongside the first few weeks of research, these will likely be used
this semester.

The data pre-processing contains code to initially extract the R peaks and labels from the annotations on the database. This was followed
by de-noising using DWT Transforms, these were used on the CNN as images.

The R-Peak detection we used was the Pan-Tompkins Algorithm.

For the CNN we then segmented the beats. For the RNN we found the corresponding RR intervals.

The CWT's were done in MATLAB however we weren't sure how to transfer MATLAB files over to GITHUB so we just pasted the code into the file in the MATLAB folder. Fortunately the code is very short so shouldn't be an issue.
