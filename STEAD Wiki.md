# STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI
STEAD is a high-quality, large-scale, and global data set of local earthquake and non-earthquake signals recorded by seismic instruments. 
The data set in its current state contains two categories:  
(1) local earthquake waveforms (recorded at “local” distances within 350 km of earthquakes)   
(2) seismic noise waveforms that are free of earthquake signals. Together these data comprise ~1.2 million time series or more than 19,000 hours of seismic signal recordings. 
Constructing such a large-scale database with reliable labels is a challenging task. 

Here, we present the properties of the data set, describe the data collection, quality control procedures, and processing steps we undertook to insure accurate labeling, 
and discuss potential applications.




## Handy Resources:
-	[STEAD Tutorial Notebook]()
-	[STEAD GitHub Page](https://github.com/smousavi05/STEAD)
-	[STEAD Paper](https://ieeexplore.ieee.org/document/8871127)
-	[STEAD Earth-ML Presentation – Part I]( https://docs.google.com/presentation/d/13jhe42NJQn1QsyUXNXp5QWaqLwrCthfaPavGnTvjPZA/edit?usp=sharing) 
-	[STEAD Earth-ML Presentation – Part II]( https://docs.google.com/presentation/d/1QU-ae8mH4veYRjPqmNyOnXdOVa_A3ul5EYa7Y2IEhRM/edit?usp=sharing)




## Construction of STEAD:

The way the STEAD is constructed is a very crucial part of comprehending what this dataset able to achieve. 
STEAD was not only capable of ensembling a huge dataset that constitutes more than 4 million phase arrival times of earthquake waveforms recorded by 3-component stations, 
but it also a highly dependable dataset that has sorted out signals. Here are the methods they used to classify and create an accurate dataset:

- Waveforms:
  -	To ensure every recorded waveform only includes one earthquake signal, STEAD used a fixed window(1 minute) around the phase arrival times. 
  -	They chose to start the window 5 to 10 seconds before the P arrival and end it 5 seconds after the S arrival, shortest. 
  -	All waveforms detrended and resampled to 100 Hz.
  -	STEAD included additional labels like the end of earthquake signals. 
  -	Estimation of the end of earthquake signal based on the time series envelope and snr measured separately for each component as:
  -	Most of the seismograms have snr between 10 and 40 decibels. The snr can be used to identify data with one or two faulty channels or to select high-quality waveforms for tasks that are sensitive to waveform quality.

- Errors:
  -	STEAD identifies a couple of errors and tries to estimate uncertainty levels. Most of them due to the lack of sensitivity of current detection algorithms or some preferences that network operators did when recording earthquakes. These errors are: 
    1.	Earthquake characterization errors: errors in location, depth, origin time, and magnitude estimates of the earthquakes and can be due to errors in the arrival time picking, inaccurate velocity models, non-robust algorithms, number of recording stations, etc.
    2.	STEAD lays out uncertainty levels in location, depth, origin time errors, and measures the quality of reported parameters. 
        -	The smaller the source_gap_deg, the more reliable is the calculated horizontal position of the earthquake.
        -	source_horizontal_uncertainty varies from about 100m horizontally for the best-located events to tens of km for global events.
    3.	Errors in arrival time picks: errors that are originating from inaccurate arrival time estimation or human errors in manual picks.
        -	To replace the theoretical arrival times with more accurate picks, STEAD used PhaseNet, a deep-learning-based phase picker. They also used CRED to identify traces with no earthquake or with more than one earthquake.
        -	Using these algorithms, STEAD detected uncatalogued earthquake signals and incorrect labels. These flaws made them reduce the size of the original waveform data set by 8%.
        -	To ensure that the noise traces do not contain earthquake signal, they applied the same method of pre-processing and post-processing.




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

-	35 EQ attributes (labels) for each earthquake seismogram:    
  1.	back_azimuth_deg 
  2.	coda_end_sample 
  3.	network_code 
  4.	p_arrival_sample 
  5.	p_status 
  6.	p_travel_sec 
  7.	p_weight 
  8.	receiver_code A16
  9.	receiver_elevation_m 
  10.	receiver_latitude 
  11.	receiver_longitude 
  12.	receiver_type 
  13.	s_arrival_sample 
  14.	s_status 
  15.	s_weight 
  16.	snr_db 
  17.	source_depth_km 
  18.	source_depth_uncertainty_km 
  19.	source_distance_deg 
  20.	source_distance_km 
  21.	source_error_sec 
  22.	source_gap_deg
  23.	source_horizontal_uncertainty_km 
  24.	source_id 
  25.	source_latitude 
  26.	source_longitude 
  27.	source_magnitude 
  28.	source_magnitude_author 
  29.	source_magnitude_type 
  30.	source_mechanism_strike_dip_rake
  31.	source_origin_time
  32.	source_origin_uncertainty_sec 
  33.	trace_category 
  34.	trace_name 
  35.	trace_start_time 

    -	Station Information:
    -	Earthquake Characteristics:
    -	Recorded Signal Information:
        -	..
        
        
-	8 Non-EQ attributes (labels) for each noise seismogram:   
    -	Recording Instrument Information:
        -	..

-	Others:


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
        -	The time of first sample is given by trace_start_time in UTC, which is randomly selected to be between 5 and 10 seconds prior to the P-arrival time. 
-	Most of the seismograms have SNR between 10 and 40 decibels




BACKLOG:
-	*source_id*:
-	A unique identification number provided by monitoring network that can be used to retrieve the wave- forms and metadata from established data centers.
-	*trace_name*:
-	A unique name containing station, network, recording time, and category code (‘‘EV’’ for earthquake and ‘‘NO’’ for noise data).

Questions:
-	How to extend STEAD with more seismograms?
-	Advantages of STEAD over other present datasets?
-	How to improve STEAD?
-	STEAD Weaknesses?
///


