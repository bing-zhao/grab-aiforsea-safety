# Dangerous Driving Classification

This repo is for submission to Grab AI Challenge for SEA on Safety.

## Introduction
On average, 1.2 million people are killed per year due to road accidents, according to a report on road safety by the World Health Organization (WHO, 2015). Provinding safety awarenesss for drivers during their trips is an effective approach to prevent such accidents. To build such intelligence, the first step is to collect driving behavior data. Nowadays, instead of installing extra devices, smart phones integrated with a set of embedded sesnors have become an economical alternative for data collection. While driving, telematics data can be eaisly collected and analyzed to detect if a driver is driving dangerously and it is possible to provide feedback to the driver in real time. This project aims to use machine learning for dangerous driving detection based on telematic data.

## Installation and Configuration
The code is developed and tested in Windows 10 with Python 3.6 under Anaconda (1.9.7) Environment. 
* The version of major libraries / packages used are listed below,
  - `numpy == 1.16.3`
  - `pandas == 0.24.0`
  - `matplotlib == 3.0.2`
  - `sklearn == 0.20.2`
  - `xgboost == 0.82`
* The folder structure is arranged in the following manner,
  - `.\srcs\`: contains the source code, environment configuration and jupyter-notebook describing the details
  - `.\model\`: save pickle files required by pre-processing, and the final trained model
  - `.\test\`: sample test input data file and output prediction results


## Execute Prediction Model for Testing
### Procedures
To execute the pre-trained predictive model, please follow the steps below:  
1. Change working directory to `./srcs/`
2. Execute the following line in command window: <br>
`python main_prediction.py --input path_to_test.csv --out path_of_prediction.csv` 

where, 
* `path_to_test.csv`: path to the test data, with default at `'../test/test.csv'`
* `path_of_prediction.csv`: path of the output prediction result, with default at `'../test/my_prediction.csv'` 

### Format of Prediction Results
The final prediction results are stored in a `.csv` file following the example label file with two columns [`bookingID`, `prediction`].


## Description of Methodology
### Pre-Processing Steps
The following pre-processing steps are performed on the given dataset,
* Removal of bookingID with multiple labels
* Removal of outlier values using IQR method 

### Rationale of Feature Engineering
In this project, both time-domain and frequency-domain features are extracted by following the metrics utilized by Lu et al. (2018) for vehicle mode and activity type detection using accelerator data of smartphones. The features being extracted includes:
* static metrics: mean, variance, standard deviation
* time domain differences: max - min, zero-crossings, cross-correlation, peak to average ratio (PAR), signal magnitude area  (SMA), signal vector magnitude (SVM), and differential signal vector magnitude (DSVM)
* frequency domain: spectral energy, and entropy
* Hjorth parameters: activity, mobility and complexity

and each of these features reflects certain aspects of the driving behavior. In addition, extra features are created to
* record the number of invalid values filtered by the outlier removal process. As abnormal fluctuation of the GPS sensor data might indicate the vehicle had entered regions with complex road environment and thus affect the driving behavior
* calculate the trip distance by integrating speed over time.
* calculate the number of zero-crossings of acceleration per time or distance (which indicates the frequency of acceleration or deceleration)
* calculate the number of zero-crossings of gyroscope's angular velocity per time or distance (which indicates the frequency of turning left/right or changing lanes)

## Summary of Model Performance and Insights
* The final XGBoost model achieved a ROC-AUC of 0.740 on training data and **0.733** for test data. 
* In general, the driving behavior has been found to be relatively more dangerous for 
    - longer trip duration: fatigue driving
    - more turning: complex route
    - higher speed: less reaction time
* In particular, the trip duration (`second`) has been found to be the most importance feature, which suggests that fatigue driving may leads to dangerous driving behavior. Thus, it is worthwhile to develop assistance driving system that provide warning to drivers during long trips or alert system to keep drivers focus their attention on the road.  

## JupyterNotebook
For more detailed steps on data processing and model development, please refer to the jupyter-notebook stored at <br> 
`./srcs/grab-safety-notebook.ipynb`

## Reference: 
Lu, D. N., Nguyen, D. N., Nguyen, T. H., & Nguyen, H. N. (2018). Vehicle mode and driving activity detection based on analyzing sensor data of smartphones. Sensors, 18(4), 1036.