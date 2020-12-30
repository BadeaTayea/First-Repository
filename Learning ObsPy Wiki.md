# Learning ObsPy:

## Aim:
The purpose of this page is to:
-	explore ObsPy, a Python library built for processing seismological data.
-	cover and apply methods and modules within ObsPy for analyzing real seismic data.
-	check overall understanding through providing two tasks along with sample solution notebooks.

## Quick guide to using this page:
-	Go over the tutorials listed below.
-	Read documentations of modules and functions on [ObsPy](https://docs.obspy.org/) along the way and whenever anything seems unclear.
-	Try solving the two tasks at the bottom and checking up your performance through the provided solutions notebooks.

## Introduction to ObsPy:
ObsPy is an open-source project dedicated to provide a Python framework for processing seismological data. 
It provides parsers for common file formats and seismological signal processing routines which allow the manipulation of seismological time series. 
The goal of the ObsPy project is to facilitate rapid application development for seismology. 

## Data in Seismology:
In seismology we generally distinguish between three separate types of data:
1.	**Waveform Data:** The actual seismic waveforms as time series (amplitude vs time graphs).
2.	**Station Data:** Information about the stations' operators, geographical locations, and the instrument's responses.
3.	**Event Data:** Information about earthquakes.

### Links to Silivri and Random Dataset: ####### Move Down 
-	[Datasets](https://drive.google.com/drive/folders/1sqXPkn9c_R9OIuD29r4w8MTBuHbth3jN?usp=sharing)###############
-	[Silivri Data]( https://drive.google.com/drive/folders/12hnk3YnKKY0n16ruvzAKAwbmHv-0LiHW)

## Downloading Data form KOERI:
-	[KOERI and EIDA Data Archives](http://eida.koeri.boun.edu.tr/webinterface/)
-	Link to a video in Drive for Download Tutorial  ############################################# 


## Tutorials by Earth-ML :
1. [Introduction to ObsPy Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/ObsPy_Tutorial_03_07_20_Copy1.ipynb): 
    -	Introduction to a few keywords and concepts (Stream – Trace – UTC Date Time - )
    -	Installing ObsPy via Pip on Jupyter Notebooks
    -	Reading seismograms
    -	Plotting earthquake waveforms
    -	Getting data by using several web-services
    -	Merging and Downsampling Seismograms
    -	Introduction to additional types of plots (Beachball, Spectrograms)

2. [Filters Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/Filters_Koray.ipynb)
    -	Introduction to different types of filters (Highpass-Lowpass) 
    -	Brief mathematical overview of filters 

3. [Instrumental Response Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/Instrument_Response_23July2020.ipynb) 
    -	Introduction to instrumental response of seismic sensors.
    -	Computing distance between earthquake and station.
    -	Reading inventory (StationXML) files of earthquakes.
    -	Plotting instrument response of stations.
    -	Plotting Displacement, Velocity, and Acceleration Graphs of ground motion using StationXML file. 
    -	Removing instrument response from earthquake waveforms using .remove_response module 
    -	Introduction to tapers and applying them in preprocessing earthquake waveforms

4. [Seismic Noise-EDA: First Pass Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/NoisedataEDAfirstpass_15-07-2020-Copy2.ipynb) 
    -	Introduction to Probabilistic Power Spectral Density (PPSD) plots
    -	Plotting PPSD of different traces
    -	Plotting Spectrograms 
    -	Investigation on the effect of volcanic micro-seismic and sea-ice noise on earthquake waveforms

5. [Seismic Noise-EDA Examples Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/NoisedataEDAfirstpass_15-07-2020-Copy2.ipynb): 
    -	More examples of PPSD plots of different traces
    -	More examples of Spectrograms plots of different traces

6. [Wavelet Transforms Notebook](https://github.com/boun-earth-ml/research/blob/cd453f25761c274c63c8f288820f7616e214764b/Tutorials/Updated__Wavelet%20Transforms%20(WT)%20of%20Seismograms%20-%20EDA%20.ipynb) 
    -	Introduction to Wavelet Transforms 
    -	Classes of Wavelet Transforms (Continuous and Discrete Wavelet Transforms)
    -	Computing Wavelet Transforms of earthquake waveforms using different methods and libraries (ObsPy – SciPy – PyWavelets)
    -	Application of different preprocessing techniques (Filtering – Tapering – Detrending – Instrument Response Removal)
    -	Plotting Spectrograms and Scalograms processed waveforms
    -	Plotting using *.imshow* module  
    -	Extra Examples




---


## Tasks & Practice:

### Task 1: 
**<ins>Prerequisites (Tutorial Notebooks):</ins>**
-	Introduction to ObsPy  
-	Filters .
-	Instrumental Response

**<ins>Task Instructions:</ins>**
-	Select any available day to read the [Silivri Data]( https://drive.google.com/drive/folders/12hnk3YnKKY0n16ruvzAKAwbmHv-0LiHW) from.
-	Merge all 3 channels (HHN, HHE, HHZ) given for the selected date, and plot resulting stream.
-	Create one-day plots for each of the three channels.
-	Select a single trace to work on, and use .slice module to slice a single trace (one of three channels) into a 5 minute duration window. Specify the dates with the UTCDateTime module. Plot the sliced trace. Advice: Work on separate copies of the trace when applying the filters at first to keep original raw trace unsliced. 
-	Apply different kinds of filtering (Highpass-Lowpass-Bandpass) over the sliced trace. Visualize and compare the effects of each of the filters with the raw trace using multiple subplots. 
-	Remove instrument response from the sliced trace (Use *slvtbefore.xml* file as inventory data).
-	Again, visually compare the differences between the processed trace and the original raw trace using multiple subplots.
-	
** Maybe specify working on a single trace…
** Review after being done with notebook 

**<ins>Expected Learning Outcomes:</ins>**
-	.
-	.
-	
**<ins>Solution Notebook:</ins>**
[Solution Notebook]


Task 2: 
Prerequisites (Tutorial Notebooks):
-	Seismic Noise-EDA: First Pass 
-	Seismic Noise-EDA Examples 
-	Wavelet Transforms
Task Instructions:
-	Download mseed/xml data from KOERI (check out the Tutorial Video if you still haven’t!).
-	Read the data downloaded in the previous step, and check out the time duration window of the data you’re reading. For convenience, make sure your data does not exceed 5 minutes (.slice module could again be useful here.).
-	Apply different kinds of wavelet transforms (using ObsPy and/or PyWavelets Packages) over the sliced dataset, and visually compare the results with the raw dataset by plotting. 
-	Plot the spectrogram of the processed dataset. Try gaining some insight from the plot. 
-	Plot the PPSD of the processed dataset. Again, try gaining some insight from the plot. 
-	For better understanding and additional practice, try downloading data from different locations around the world (Advice: Try comparing data of noisy and silent locations. Examples: Paris(France)/ Reykjavík(Iceland))

Expected Learning Outcomes:
-	

Solutions:
[Solution Notebook]


Additional Comments:
-	The tasks cover only the basics to be learned from the tutorials …
-	Given that you have followed the …
-	STA/LTA
-	Template Matching 
-	Silivri Reports
