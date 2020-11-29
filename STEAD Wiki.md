# STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI
STEAD is a high quality, large-scale, and global dataset of both local earthquake and noise signals recorded by seismic instruments. The data set offers a a valuable training data set that could be utilized to increase AI-based algorithms’ efficiency and accuracy in denoising, detection, phase picking, and classification/discrimination of seismic signals. The current data set is marked with having well-established control algorithms to check and eliminate inaccurately labeled signals.

The data set in its current state contains two categories:    
1. Local earthquake waveforms (recorded at “local” distances within 350 km of earthquakes)     
2. Seismic noise waveforms that are free of earthquake signals. Together these data comprise ~1.2 million time series or more than 19,000 hours of seismic signal recordings. 

This page presents the properties of the data set and describe how STEAD is constructed.


## Tutorial Notebook:
The [STEAD Tutorial Notebook]() has been designed for the purpose of:
- exploring how to access, select, and display earthquake and non-earthquak (noise) waveforms within STEAD.
- covering how to convert raw waveforms within STEAD to velocity, acceleration, and displacement.
- checking on the functioning and output of the python code presented in STEAD GitHub Page.



## Handy Resources:
-	[STEAD GitHub Page](https://github.com/smousavi05/STEAD)
-	[STEAD Paper](https://ieeexplore.ieee.org/document/8871127)
-	[STEAD Earth-ML Presentation – Part I]( https://docs.google.com/presentation/d/13jhe42NJQn1QsyUXNXp5QWaqLwrCthfaPavGnTvjPZA/edit?usp=sharing) 
-	[STEAD Earth-ML Presentation – Part II]( https://docs.google.com/presentation/d/1QU-ae8mH4veYRjPqmNyOnXdOVa_A3ul5EYa7Y2IEhRM/edit?usp=sharing)

---

## STEAD in Numbers:

### Two Main Classes of Signals Recorded by Seismic Instruments:
-	**EQ Signals:**
      -	One Category of Local EQs
      -	~1,050,000 Three-Component Seismograms (each 1 minute long)
      -	Each seismogram is associated with ~ 450,000 EQs 
      -	EQs occurred between January 1984 and August 2018
      -	EQs recorded by 2,613 seismometers
      -	EQs recorded at local distances (within 350 km of the earthquakes)
      
-	**Non-EQ Signals:**
      -	One Category of Seismic Noise
      -	~100,000 Samples 
 
 
### Seismic Data Properties:
-	Individual NumPy arrays containing three waveforms:
    -	Each waveform has 6000 samples
    - Each waveform is associated with 60 seconds of ground motion recorded in east-west, north-south, and vertical directions
    -	More than 6200 waveforms contain information about the earthquake focal mechanisms.

-	35 EQ attributes (labels) for each earthquake seismogram, including information on:    
    - Station
    - Earthquake Characteristics  
    - Recorded Signal
    - Seismogram Identification:
      - *source_id*: A unique identification number provided by monitoring network that can be used to retrieve the waveforms and metadata from established data centers.  
      - *trace_name*: A unique name containing station, network, recording time, and category code (‘‘EV’’ for earthquake and ‘‘NO’’ for noise data).  
      - *source_magnitude_author*: A unique name of the institute that calculated the magnitude.  
      - *network_code*: A unique code for the seismic monitoring network to which the recording instrument belongs. It can be used for retrieving either the waveform or the metadata directly from the monitoring network.  
        
        
-	8 Non-EQ attributes (labels) for each noise seismogram:   
    -	Recording Instrument Information
    - Trace ID



### Magnitude Information:
-	Magnitudes Range: [0.5, 7.9]
-	Small earthquakes (magnitudes < 2.5) comprise most of the data set. 
-	Magnitudes have been reported in 23 different magnitude scales where local (ml) and duration (md) magnitudes are the majority.
    -	ml, mb, md, mw, ms, mwr, mb_lg, mn, mpv, mlg, mwc, mc, mg, mh, mlr, mww, mpva, mbr, mblg, mwb, mlv, h, m, and mdl scales.
-	Uncertainties for magnitude estimations have not been reported and only in ~24 % of the cases, the name of institute that calculated the magnitude (source_magnitude_author) were reported and have been provided.


### P,S Phase Arrival Times:
-	Three types of P,S arrival statuses in the data set:
    -	‘‘Manual’’ picks hand-picked by human analysts (compromise 70% of the dataset.)
    -	‘‘Automatic’’ picks are those measured by automatic algorithms
    -	‘‘Autopicker’’ are arrival times determined using our AI-based model in this study.
-	A measure of uncertainties in arrival time picks, a weight (a number between 0 and 1) is provided for most cases.


### Instrumentation, Seismic Networks, and Seismograms:
-	Instruments belong to 144 seismic networks operated at local, regional, and global scales by different national and international agencies.
-	Data is recorded by only 7 types of instruments:
    -	99.5% are either high-gain broad band or extremely short period.
-	**All seismograms (earthquake and non-earthquake) are:** 
    -	Three-component
    -	Resampled to 100 HZ
    -	60 second (6000 samples) in duration
        -	The time of first sample is given by *trace_start_time* in UTC, which is randomly selected to be between 5 and 10 seconds prior to the P-arrival time. 
-	Most of the seismograms have SNR between 10 and 40 decibels



---



## **Construction of STEAD**
The way STEAD is constructed is a very crucial part of comprehending what this dataset is able to achieve. STEAD is not only capable of ensembling a huge dataset that constitutes more than 4 million phase arrival times of earthquake waveforms recorded by 3-component stations, but it also is a highly dependable dataset that has sorted out signals. Here are the methods the authors of STEAD used to classify and create an accurate dataset:
### Waveforms:
* To ensure every recorded waveform only includes one earthquake signal, STEAD uses a fixed window(1 minute) around the phase arrival times.
* The window is chosen 5 to 10 seconds before the P arrival and end it 5 seconds after the S arrival, shortest.
* All waveforms are detrended and resampled to 100 Hz.
* STEAD includes additional labels such as the end of earthquake signals.
* Estimation of the end of earthquake signal is based on the time series envelope and snr measured separately for each component as:
Most of the seismograms have snr between 10 and 40 decibels. The snr can be used to identify data with one or two faulty channels or to select high-quality waveforms for tasks that are sensitive to waveform quality.
### Errors:
STEAD identifies a couple of errors and tries to estimate uncertainty levels. Most of them are due to the lack of sensitivity of current detection algorithms or some preferences that network operators did when recording earthquakes. These errors are:
* **Earthquake characterization errors:** Errors in location, depth, origin time, and magnitude estimates of the earthquakes and can be due to errors in the arrival time picking, inaccurate velocity models, non-robust algorithms, number of recording stations, etc.
    1.  STEAD lays out uncertainty levels in location, depth, origin time errors, and measures the quality of reported parameters.
    1. The smaller the ***source_gap_deg***, the more reliable is the calculated horizontal position of the earthquake.
     1. ***source_horizontal_uncertainty*** varies from about 100m horizontally for the best-located events to tens of km for global events.
   1. Due to the different depth determination methods, the error bars of the source depth can be huge. For example, for the shallow areas default depth is often used as 33 km, but in shallower areas, like mid-ocean ridges, the default depth is 5 or 10 km. Operators used default depths when depth is poorly constrained by seismic data.
* **Errors in arrival time picks:** errors that are originating from inaccurate arrival time estimation or human errors in manual picks.
   1. To replace the theoretical arrival times with more accurate picks, STEAD used PhaseNet, a deep-learning-based phase picker. CRED is also used to identify traces with no earthquake or with more than one earthquake.
    1. Using these algorithms, STEAD detected uncatalogued earthquake signals and incorrect labels. These flaws made them reduce the size of the original waveform data set by 8%.
   1. To ensure that the noise traces do not contain earthquake signal, the same method of pre-processing and post-processing is applied.
   1. Unfortunately, the algorithms that are used in some of the seismic networks lack sensitivity. Overall, STEAD tries to ensure that there is no unrecorded earthquake in the catalog by applying this procedure.




---

Although STEAD can be used in various fields of applications, it was designed as a valuable tool for deep-learning algorithms to process and characterize earthquakes. In recent studies, one can observe that machine learning models can outperform classical algorithms. However, these models must be trained on reliable data. The existence of a data set like STEAD can accelerate the advancement of these algorithms significantly. Future models can foster building more accurate phase pickers, promising to offer a solution to the problem of classification of seismic signals, to rapidly estimate the magnitude, distance, and depth of earthquakes which may improve early warning systems, and to directly determine the earthquake locations.  

The authors of STEAD are trying to expand this data set to regional and teleseismic(> 2000 km distance) earthquake seismograms and diversify the data with the seismic waves generated by explosions, volcanoes, planes, wind, and traffic. The utmost goal remains to prepare a data set capable of building high-precision models and accelerating develpments in seismology.

