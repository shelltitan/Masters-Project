# Masters-Project
Automatic Detection of Cardiac Arrhythmias

The following list is in order of completion as well as brief descriptions.

The Poincare plot codes were done at the very start of the project alongside the first few weeks of research, these will likely be used
this semester.

The data pre-processing contains code to initially extract the R peaks and labels from the annotations on the database. This was followed
by de-noising using DWT Transforms.

The R-Peak detection we used was the Pan-Tompkins Algorithm.

For the CNN:
We then segmented the beats followed by CWT transformations in MATLAB. We weren't sure how to add MATLAB files into GITHUB but fortunately, the program was really short so we just pasted that over into a file in the MATLAB folder. These images were then put into random samples and converted into arrays. These were then used in the CNN as well as more code to continue training the CNN.

For the RNN:
We calculated the RR intervals. We then used sliding windows over the data that saved all the RR information and labels corresponding to that window. These were then inputted into the RNN in batches of 1024. No code was needed for continuing training as the RNN was quick enough to do in one session.
