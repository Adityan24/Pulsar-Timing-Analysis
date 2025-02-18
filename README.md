Gravitational Wave Analysis using Pulsar Timing Array Data
This repository contains the source code used in the research paper:
Tracing Gravitational Waves by integrating wavelet ,PCA and clustering analysis using Pulsar Timing Array Data
 
Project Overview
This project explores the detection of gravitational waves through the analysis of pulsar timing array (PTA) data, employing advanced signal processing and machine learning techniques. The primary objective is to identify gravitational wave signatures by:
•	Wavelet Transformation: Denoising and extracting relevant signal features across multiple time-frequency scales.
•	Principal Component Analysis (PCA): Reducing the dimensionality of the extracted features while retaining critical information.
•	Clustering Algorithms: Grouping similar signal patterns to detect gravitational wave events.
This approach provides a robust method for isolating weak gravitational signals from noise, leveraging pulsar timing data.
 
Code Structure
•	main.py: Main script that integrates wavelet transformation, PCA, and clustering analysis.
Key Functions:
•	Data Loading: Reads PTA observational data.
•	Wavelet Decomposition: Applies pywt for multi-level decomposition and noise reduction.
•	Dimensionality Reduction: Uses sklearn PCA to project data onto principal components.
•	Clustering: Employs K-Means and hierarchical clustering to classify signal patterns.
•	Visualization: Plots wavelet coefficients, PCA components, and clustering results using matplotlib.
 
Installation and Dependencies
Ensure Python 3 is installed. Install required libraries using:

pip install numpy scipy scikit-learn pywt matplotlib
 
Usage Guide
To execute the analysis, run:
python main.py
The script performs the following:
•	Preprocesses raw PTA data.
•	Applies wavelet transformation to denoise signals.
•	Reduces dimensions using PCA.
•	Clusters signals to detect potential gravitational waves.
•	Generates visual plots for analysis.
 
Outputs and Results
•	Wavelet-transformed signals: Enhanced signals with reduced noise.
•	PCA components: Principal features representing the signal.
•	Clusters: Groups indicating potential gravitational wave signatures.
 
Research Context
This project supports the research presented in the paper:
Tracing Gravitational Waves by integrating wavelet, PCA, and clustering analysis using Pulsar Timing Array Data.
The methodology bridges the gap between theoretical astrophysics and computational data analysis, providing a toolset for future gravitational wave studies.
 
Citation
If you use this code, please cite the research paper:
Tracing Gravitational Waves by integrating wavelet, PCA, and clustering analysis using Pulsar Timing Array Data. (communicated to Astrophysics and Space Science Journal, Springer )
