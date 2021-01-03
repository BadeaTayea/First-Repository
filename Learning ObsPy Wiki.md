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

## Downloading Data form KOERI:
Use the following link to access global earthquake waveform datasets:
-	[KOERI and EIDA Data Archives](http://eida.koeri.boun.edu.tr/webinterface/)

### Video Tutorial:
The following is a quick video tutorial that has been prepared to offer a visual guide to downloading data from KOERI:  
-   [Downloading Data form KOERI/EIDA Tutorial Video - Drive Link]()

### Links to Silivri and Random Datasets: 
-	[Silivri Data]( https://drive.google.com/drive/folders/12hnk3YnKKY0n16ruvzAKAwbmHv-0LiHW)
-	[Some Random Datasets](https://drive.google.com/drive/folders/1sqXPkn9c_R9OIuD29r4w8MTBuHbth3jN?usp=sharing)


## Tutorials by Earth-ML :
1. [Introduction to ObsPy - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/ObsPy_Tutorial_03_07_20_Copy1.ipynb): 
    -	Introduction to key terms and concepts (Stream – Trace – UTC Date Time)
    -	Installing ObsPy via Pip on Jupyter Notebooks
    -	Reading seismograms
    -	Plotting earthquake waveforms
    -	Acquiring data by using several web-services
    -	Merging and downsampling seismograms
    -	Introduction to new types of plots (Beachball, Spectrograms)

2. [Filters - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/Filters_Koray.ipynb):
    -	Introduction to different types of filters (Highpass-Lowpass) 
    -	Brief mathematical overview of filters 

3. [Instrumental Response - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/Instrument_Response_23July2020.ipynb):
    -	Introduction to instrumental response of seismic sensors
    -	Computing distance between earthquake and station
    -	Reading inventory (StationXML) files of earthquake waveforms
    -	Plotting instrument response of stations
    -	Plotting Displacement, Velocity, and Acceleration graphs of ground motion using StationXML files 
    -	Removing instrument response from earthquake waveforms using *.remove_response()* module 
    -	Introduction to tapers and applying them in preprocessing earthquake waveforms

4. [Seismic Noise-EDA: First Pass - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/NoisedataEDAfirstpass_15-07-2020-Copy2.ipynb):
    -	Introduction to Probabilistic Power Spectral Density (PPSD) plots
    -	Plotting PPSDs of different traces
    -	Plotting Spectrograms 
    -	Investigation on the effect of volcanic micro-seismic and sea-ice noise on earthquake waveforms

5. [Seismic Noise-EDA Examples - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/NoisedataEDAfirstpass_15-07-2020-Copy2.ipynb): 
    -	More examples of PPSD plots of different traces
    -	More examples of Spectrogram plots of different traces

6. [Wavelet Transforms - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/cd453f25761c274c63c8f288820f7616e214764b/Tutorials/Updated__Wavelet%20Transforms%20(WT)%20of%20Seismograms%20-%20EDA%20.ipynb) 
    -	Introduction to Wavelet Transforms 
    -	Classes of Wavelet Transforms (Continuous and Discrete Wavelet Transforms)
    -	Computing Wavelet Transforms of earthquake waveforms using different methods and libraries (ObsPy – SciPy – PyWavelets)
    -	Application of different preprocessing techniques (Filtering – Tapering – Detrending – Instrument Response Removal)
    -	Plotting Spectrograms and Scalograms of pre-processed waveforms
    -	Plotting using *.imshow()* module  
    -	Extra Examples




---


## Tasks & Practice:

### Task 1: 
**<ins>Prerequisites (Tutorial Notebooks):</ins>**  
-	[Introduction to ObsPy - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/ObsPy_Tutorial_03_07_20_Copy1.ipynb) 
-	[Filters - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/Filters_Koray.ipynb)
-	[Instrumental Response - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/Instrument_Response_23July2020.ipynb)

**<ins>Task Instructions:</ins>**  
1.	Select any available day to read the [Silivri Data]( https://drive.google.com/drive/folders/12hnk3YnKKY0n16ruvzAKAwbmHv-0LiHW) from.
2.	Merge all 3 given channels (HHN, HHE, HHZ) for the selected date, and plot the merged stream.
3.	Create one-day plots for each of the three channels.
4.	Optional: Apply sorting to selected stream.
5.	Optional: Use Matplotlib subplots to display all 3 traces (channels) within stream.
6.	Select a single trace to work on, and use *.slice()* module to slice a single trace (one of three channels) into a 5 minute duration window. Specify the dates using the UTCDateTime modules. 
7.	Plot the sliced trace. 
8.	Apply different kinds of filtering (Highpass-Lowpass-Bandpass) over the sliced trace. Visualize and compare the effects of each of the filters with the raw trace using multiple subplots. Advice: Work on separate copies of the trace when applying the filters to keep the original, raw trace unchanged. 
9.	Read metadata of original stream (Use *slvtbefore.xml* file as inventory data).
10.	Go over the stream’s inventory data and display inventory information of all three available channels.
11.	Optional: Plot instrument response of stream, along with the Displacement, Velocity, and Acceleration graphs of ground motion.
12.	Remove instrument response from data, and plot resulting trace.
13.	Visually compare the differences between the processed trace and the original raw trace using multiple subplots.


**<ins>Expected Learning Outcomes:</ins>**   
Going through this task, you should now be able to:
-	Read, merge, slice, and select streams/traces.
-	Plot streams, traces, and one-day graphs of traces.
-	Apply different types of filtering on waveform data and gain some insight about the changes they result in.
-	Read metadata of earthquake waveforms, as well as remove instrument response using this type of data.
-	Plot instrument response and ground motion graphs of earthquake waveforms.
-	Understand and visualize the effects of removing instrument response from waveform traces.


**<ins>Sample Solution Notebook:</ins>**    
[Learning ObsPy Part 1 - Jupyter Notebook]()


### Task 2: 
**<ins>Prerequisites (Tutorial Notebooks):**</ins>   
-	[Seismic Noise-EDA: First Pass - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/NoisedataEDAfirstpass_15-07-2020-Copy2.ipynb) 
-	[Seismic Noise-EDA Examples - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/master/Tutorials/NoisedataEDAfirstpass_15-07-2020-Copy2.ipynb)
-	[Wavelet Transforms - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/cd453f25761c274c63c8f288820f7616e214764b/Tutorials/Updated__Wavelet%20Transforms%20(WT)%20of%20Seismograms%20-%20EDA%20.ipynb)

 
**<ins>Task Instructions:**</ins>   
1.	Download mseed/xml data from KOERI (check out the [Tutorial Video]() if you still haven’t!), or use any of Silivri’s Datasets.
2.	Read the data acquired in the previous step, and select a single trace to work on. For convenience, work with data in a 5 minutes duration window (*.slice()* module could again be utilized here).
3.	Apply different preprocessing techniques on selected trace (normalization – detrending - tapering).
4.	Read metadata of original stream (*.xml* file), and use it to remove the instrument response from the selected trace.
5.	Plot the preprocessed trace using Matplotlib and ObsPy’s *.plot()* module.
6.	Plot the spectrogram of the selected trace using ObsPy’s *.spectrogram()* module.
7.	Plot the Probabilistic Power Spectral Density PPSD of the selected trace using ObsPy’s PPSD module.
8.	Start working with Wavelet Transforms on several copies of the selected trace. Apply Continuous Wavelet Transform using any of the methods specified in the [Wavelet Transforms - Jupyter Notebook](https://github.com/boun-earth-ml/research/blob/cd453f25761c274c63c8f288820f7616e214764b/Tutorials/Updated__Wavelet%20Transforms%20(WT)%20of%20Seismograms%20-%20EDA%20.ipynb) (ObsPy – SciPy – Pywavelets).
9.	Apply Discrete Wavelet Transform on selected trace using Pywavelets.
10.	Optional: For better understanding and additional practice, try downloading and working with data from different locations around the world (Advice: Try comparing data of noisy and silent locations. Examples: Paris(France)/ Reykjavík(Iceland), respectively).



**<ins>Expected Learning Outcomes:**</ins>  
-	Downloading data from KOERI. 
-	Plotting spectrograms of earthquake waveforms.
-	Plotting PPSDs of earthquake waveforms.
-	Applying CWTs and DWTs on preprocessed waveform traces.


**<ins>Sample Solution Notebook:</ins>**   
[Learning ObsPy Part 2 - Jupyter Notebook]()


## Additional Comments:  
This page has been created for the purpose of introducing key concepts and commands within ObsPy in an organized fashion. Only some of the tutorials prepared by Earth-ML have been linked to this page. For learning more about ObsPy, check out all the tutorials prepared on this [Tutorials Page](https://github.com/boun-earth-ml/research/tree/master/Tutorials). Good Luck!


