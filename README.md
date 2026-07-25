# Canonical Correlation Analysis (CCA) Toolkit
Meysam Amirsardari (Univ. of Maryland) and 
Malcolm Slaney (Stanford CCRMA and ICSI, Berkeley)

Our goal is to provide a production quality, reference implementation of
canonical correlation analysis (CCA). In our case, we are interested in finding
a linear model that best connects auditory stimulation to (EEG) brain waves. 
As shown in the figure below, this can be done in several different fashions:
including forward (predict EEG from audio) and backward (predict audio from EEG).
But the paper by de Cheveigne shows that we get the best model if we rotate
the audio and the EEG signals, each in their own canonical directions, to 
find a new subspace that maximizes the resulting correlation.
CCA is not limited to brain data, and can be used to find a match between 
any two related data [for example Slaney/Covell]

<img src="images/CCA Overview.png" alt="Overview if CCA and Alternatives" 
  width="300" height="200">


# References
de Cheveigne

Slaney, Covell. FaceSync