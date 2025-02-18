# Code from /mnt/data/hde.py
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ---------------------- Data I/O ----------------------------
def read_pulsar_data(directory):
    """
    Read pulsar PTA data from text files in the specified directory.
    Each text file is assumed to contain a numerical array of timing residuals.
    The file name (without extension) is used as the pulsar name.
    """
    pulsar_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Use the filename (without extension) as the pulsar name.
            pulsar_name = os.path.splitext(filename)[0]
            data = np.loadtxt(os.path.join(directory, filename))
            pulsar_data[pulsar_name] = data
    return pulsar_data

# ---------------------- Known Pulsar Positions ----------------------------
def get_positions():
    """
    Return a dictionary with the known sky positions for each PTA pulsar.
    The positions are given as (RA, Dec) in radians.
    
    (The following coordinates are approximate.)
      - J1643-1224: RA = 16:43:37  (250.9035°), DEC = -12:24:30  (-12.4083°)
      - J1713+0747: RA = 17:13:49.53 (258.456°),  DEC = +07:47:37.9 (7.7939°)
      - J1909-3744: RA = 19:09:47.44 (287.448°),  DEC = -37:44:14.5 (-37.7374°)
      - J1939+2134: RA = 19:39:38.56 (294.910°),  DEC = +21:34:59.1 (21.5831°)
      - J2145-0750: RA = 21:45:50.45 (326.460°),  DEC = -07:50:18.3 (-7.8384°)
    """
    positions = {
        "J1643-1224": (np.deg2rad(250.9035), np.deg2rad(-12.4083)),
        "J1713+0747": (np.deg2rad(258.456),  np.deg2rad(7.7939)),
        "J1909-3744": (np.deg2rad(287.448), np.deg2rad(-37.7374)),
        "J1939+2134": (np.deg2rad(294.910), np.deg2rad(21.5831)),
        "J2145-0750": (np.deg2rad(326.460), np.deg2rad(-7.8384))
    }
    return positions

def angular_separation(pos1, pos2):
    """
    Compute the angular separation (in radians) between two sky positions.
    pos1 and pos2 are tuples (RA, Dec) in radians.
    Uses the spherical cosine formula.
    """
    ra1, dec1 = pos1
    ra2, dec2 = pos2
    cos_theta = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)

# ---------------------- Hellings–Downs Curve ----------------------------
def HD_curve(theta):
    """
    Compute the standard Hellings–Downs curve for angular separation theta (radians).
    For theta = 0, define HD(0)=1.
    Avoids log(0) by handling x == 0 explicitly.
    """
    theta = np.atleast_1d(theta)
    x = (1 - np.cos(theta)) / 2.0
    hd = np.empty_like(x)
    mask = x > 0
    hd[~mask] = 1.0
    hd[mask] = 0.5 + 1.5 * x[mask] * np.log(x[mask]) - 0.25 * x[mask]
    return hd if hd.size > 1 else hd[0]

# ---------------------- Cross-correlation Analysis ----------------------------
def compute_pairwise_correlations(pulsar_data, positions):
    """
    For each unique pulsar pair, compute:
      - The angular separation (radians) based on the known positions.
      - The Pearson correlation coefficient between their timing residuals.
    
    If the time series differ in length, they are trimmed to the length of the shorter.
    Assumes the data arrays are 1D or (if multidimensional) uses the first column.
    Returns arrays of separations and correlations.
    """
    separations = []
    correlations = []
    pulsar_names = list(pulsar_data.keys())
    
    for name1, name2 in combinations(pulsar_names, 2):
        data1 = pulsar_data[name1]
        data2 = pulsar_data[name2]
        # Use first column if data are multidimensional.
        if data1.ndim > 1:
            data1 = data1[:, 0]
        if data2.ndim > 1:
            data2 = data2[:, 0]
            
        # Trim to the same length.
        min_len = min(len(data1), len(data2))
        if len(data1) != len(data2):
            data1 = data1[:min_len]
            data2 = data2[:min_len]
        
        # Compute Pearson correlation.
        r, _ = pearsonr(data1, data2)
        # Use the known positions for angular separation.
        sep = angular_separation(positions[name1], positions[name2])
        separations.append(sep)
        correlations.append(r)
    
    return np.array(separations), np.array(correlations)

# ---------------------- Binning & Fitting ----------------------------
def bin_data(separations, correlations, n_bins=10, epsilon=1e-8):
    """
    Bin the correlation values by angular separation.
    Returns bin centers, mean correlation per bin, and the standard error in each bin.
    If the standard error is zero, replace it with a small epsilon.
    """
    bins = np.linspace(0, np.pi, n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    binned_corr = np.zeros(n_bins)
    binned_err = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (separations >= bins[i]) & (separations < bins[i+1])
        if np.any(mask):
            vals = correlations[mask]
            binned_corr[i] = np.mean(vals)
            std_err = np.std(vals) / np.sqrt(len(vals))
            binned_err[i] = std_err if std_err > 0 else epsilon
        else:
            binned_corr[i] = np.nan
            binned_err[i] = np.nan
    return bin_centers, binned_corr, binned_err

def model_HD(theta, A):
    """
    Model function for fitting: a scaled Hellings–Downs curve.
    """
    return A * HD_curve(theta)
    

# ---------------------- Main Execution ----------------------------
if __name__ == '__main__':
    # Set your directory path here (where the PTA data text files are located)
    directory = "/users/adityan/downloads/ptadata"
    pulsar_data = read_pulsar_data(directory)
    
    # Get the known positions (do not simulate)
    positions = get_positions()
    
    # Compute pairwise angular separations and Pearson correlations.
    separations, correlations = compute_pairwise_correlations(pulsar_data, positions)
    
    # Bin the data by angular separation.
    bin_centers, binned_corr, binned_err = bin_data(separations, correlations, n_bins=10)
    
    # Remove any bins with no data.
    valid = ~np.isnan(binned_corr)
    bin_centers = bin_centers[valid]
    binned_corr = binned_corr[valid]
    binned_err = binned_err[valid]
    
    # Fit the binned correlation data to the model: correlation = A * HD_curve(theta)
    popt, pcov = curve_fit(model_HD, bin_centers, binned_corr, sigma=binned_err, absolute_sigma=True)
    A_fit = popt[0]
    A_err = np.sqrt(np.diag(pcov))[0]
    
    # Compute the chi-squared statistic.
    model_vals = model_HD(bin_centers, A_fit)
    chi2 = np.sum(((binned_corr - model_vals) / binned_err)**2)
    dof = len(binned_corr) - 1  # one fitted parameter
    print(f"Fitted amplitude A = {A_fit:.3f} ± {A_err:.3f}")
    print(f"Chi-squared/dof = {chi2:.2f}/{dof}")
    
    # Pearson correlation between binned data and model predictions.
    r_bin, p_val = pearsonr(binned_corr, model_vals)
    print(f"Pearson correlation (binned data vs. model): r = {r_bin:.3f} (p = {p_val:.3g})")
    
    # ---------------------- Plotting ----------------------------
    plt.figure(figsize=(8, 6))
    # Plot the binned correlations with error bars (angular separation in degrees)
    plt.errorbar(np.degrees(bin_centers), binned_corr, yerr=binned_err, fmt='o', label='Binned cross-correlation')
    
    # Fine grid for plotting curves.
    theta_fine = np.linspace(0, np.pi, 200)
    plt.plot(np.degrees(theta_fine), HD_curve(theta_fine), 'k--', label='Canonical HD curve')
    plt.plot(np.degrees(theta_fine), model_HD(theta_fine, A_fit), 'r-', label=f'Best-fit: A={A_fit:.2f}')
    
    plt.xlabel("Angular separation (deg)")
    plt.ylabel("Correlation coefficient")
    plt.title("Cross-correlation vs. Angular Separation\n(Fitting to Hellings–Downs Curve)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
from scipy.optimize import curve_fit
import emcee
import os
import pywt  # Import wavelet module

# Define power-law model
def power_law(f, A, gamma, C):
    return np.maximum(A * f ** gamma + C, 1e-20)  # Ensure positive values

# Compute PSD using Lomb-Scargle Periodogram
def compute_psd(time, residuals):
    freqs = np.logspace(-9, -6, 100)  # PTA frequency range
    psd = lombscargle(time, residuals, freqs, normalize=True)
    return freqs, np.maximum(psd, 1e-20)  # Ensure PSD is positive

# Compute PSD using Wavelet Transform
def compute_wavelet_psd(time, residuals):
    scales = np.logspace(0.1, 3, 100)  # Define wavelet scales
    coefficients, freqs = pywt.cwt(residuals, scales, 'morl', sampling_period=np.mean(np.diff(time)))
    power_spectrum = np.mean(np.abs(coefficients) ** 2, axis=1)  # Compute power spectrum
    return freqs, power_spectrum

# Save PSD data
def save_psd_data(pulsar_name, freqs, psd, output_dir="processed_data"):
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f"{pulsar_name}_psd.txt"), np.column_stack((freqs, psd)))

# Log-prior function for Bayesian inference
def log_prior(params):
    A, gamma, C = params
    if 1e-18 < A < 1e-12 and -5 < gamma < 0 and 1e-20 < C < 1e-10:
        return 0.0
    return -np.inf

# Log-likelihood function
def log_likelihood(params, f, psd):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    
    A, gamma, C = params
    model = power_law(f, A, gamma, C)
    return -0.5 * np.sum((psd - model) ** 2 / model + np.log(model))

# Function to fit the power-law model using curve_fit
def fit_power_law(freqs, psd):
    popt, pcov = curve_fit(power_law, freqs, psd, p0=[1e-15, -13/3, 1e-20], maxfev=10000)
    return popt, np.sqrt(np.diag(pcov))

# Main processing function
def process_pulsar(pulsar_name, time, residuals):
    # Compute Fourier-based PSD
    freqs_fourier, psd_fourier = compute_psd(time, residuals)
    
    # Compute Wavelet-based PSD
    freqs_wavelet, psd_wavelet = compute_wavelet_psd(time, residuals)
    
    # Fit Power-Law Model to Wavelet PSD
    params_wavelet, errors_wavelet = fit_power_law(freqs_wavelet, psd_wavelet)
    
    # Save Data
    save_psd_data(pulsar_name + "_fourier", freqs_fourier, psd_fourier)
    save_psd_data(pulsar_name + "_wavelet", freqs_wavelet, psd_wavelet)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.loglog(freqs_fourier, psd_fourier, 'r-', label='Fourier PSD')
    plt.loglog(freqs_wavelet, psd_wavelet, 'b--', label='Wavelet PSD')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.title(f'Power Spectral Density - {pulsar_name}')
    plt.savefig(f"{pulsar_name}_psd_comparison.png")
    plt.show()
    
    return params_wavelet, errors_wavelet


# Code from /mnt/data/psl_updated.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pywt
from scipy.signal import correlate, welch
import scipy.signal as signal

try:
    from skfuzzy.cluster import cmeans
except ModuleNotFoundError:
    print("Error: The 'skfuzzy' library is not installed. Please install it using 'pip install scikit-fuzzy' and try again.")
    raise

# List of dataset filenames
datasets = [
    '/Users/adityan/Downloads/InPTADRIa-main/J1643-1224.DMtimeseries_corrected.csv',
    '/Users/adityan/Downloads/InPTADRIa-main/J1713+0747.DMtimeseries_corrected.csv',
    '/Users/adityan/Downloads/InPTADRIa-main/J1909-3744.DMtimeseries_corrected.csv',
    '/Users/adityan/Downloads/InPTADRIa-main/J1939+2134.DMtimeseries_corrected.csv',
    '/Users/adityan/Downloads/InPTADRIa-main/J2145-0750.DMtimeseries_corrected.csv'
]

# Initialize variables to store combined data
all_MJD = []
all_timing_residuals = []

# Load and process each dataset
for filename in datasets:
    data = pd.read_csv(filename)
    
    # Extract relevant columns
    MJD = data['MJD'].values
    DM = data['DM'].values
    DM_uncertainty = data['DM_uncertainty'].values
    TOA_correction = data['TOA_correction'].values
    
    # Compute timing residuals
    timing_residuals = TOA_correction - (DM * DM_uncertainty)
    
    # Add synthetic noise and GW signal
    noise = 0.0001 * np.random.randn(len(timing_residuals))
    timing_residuals_noisy = timing_residuals + noise
    gw_signal = 0.0005 * np.sin(2 * np.pi * 0.01 * np.arange(len(timing_residuals)))
    timing_residuals_gw = timing_residuals_noisy + gw_signal
    
    # Store results
    all_MJD.extend(MJD)
    all_timing_residuals.extend(timing_residuals_gw)

all_MJD = np.array(all_MJD)
all_timing_residuals = np.array(all_timing_residuals)

# Step 1: PCA Analysis
data_matrix = np.column_stack((all_MJD, all_timing_residuals))
pca = PCA()
scores = pca.fit_transform(data_matrix)
explained_variance = pca.explained_variance_ratio_ * 100

# Print explained variance for first few PCs
for i, var in enumerate(explained_variance[:5], start=1):
    print(f'Principal Component {i}: {var:.2f}% variance explained')



# Step 2: Fuzzy C-Means Clustering
num_clusters = 3
pca_data = scores[:, :2].T  # Use first two principal components
cntr, u, _, _, _, _, _ = cmeans(pca_data, num_clusters, 2, 0.005, 1000)
cluster_labels = np.argmax(u, axis=0)

# Plot Fuzzy C-Means Clustering
plt.figure()
for i in range(num_clusters):
    plt.scatter(pca_data[0, cluster_labels == i], pca_data[1, cluster_labels == i], label=f'Cluster {i + 1}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Fuzzy C-Means Clustering')
plt.legend()
plt.grid()
plt.show()

# Step 3: K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(scores[:, :2])

# Plot K-Means Clustering
plt.figure()
for i in range(num_clusters):
    plt.scatter(pca_data[0, kmeans_labels == i], pca_data[1, kmeans_labels == i], label=f'Cluster {i + 1}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering')
plt.legend()
plt.grid()
plt.show()

# Step 4: Wavelet Analysis
wavelet_family = 'db4'
decomposition_level = 5
coeffs = pywt.wavedec(all_timing_residuals, wavelet_family, level=decomposition_level)

# Create heatmap for wavelet coefficients
feature_matrix = np.zeros((len(coeffs), max(len(c) for c in coeffs)))
for i, c in enumerate(coeffs):
    feature_matrix[i, :len(c)] = c

plt.figure()
plt.imshow(feature_matrix, aspect='auto', cmap='hot', origin='lower')
plt.colorbar(label='Wavelet Coefficients')
plt.xlabel('Coefficient Index')
plt.ylabel('Decomposition Levels')
plt.title('Wavelet Coefficients Heatmap')
plt.show()

# Continuous Wavelet Transform Analysis
scales = np.arange(1, 51)
coefficients, frequencies = pywt.cwt(all_timing_residuals, scales, 'cmor')

plt.figure(figsize=(12, 6))
plt.imshow(np.abs(coefficients), extent=[all_MJD[0], all_MJD[-1], scales[-1], scales[0]], 
           aspect='auto', cmap='viridis')
plt.colorbar(label='Magnitude of Coefficients')
plt.xlabel('MJD')
plt.ylabel('Scale')
plt.title('Continuous Wavelet Transform (CWT) Analysis')
plt.show()

# Generate Scaled Wavelet Analysis Plots
resolutions = [0.5, 1, 2]
for resolution in resolutions:
    adjusted_scales = scales * resolution
    adjusted_coefficients, _ = pywt.cwt(all_timing_residuals, adjusted_scales, 'cmor')

    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(adjusted_coefficients), extent=[all_MJD[0], all_MJD[-1], adjusted_scales[-1], adjusted_scales[0]], 
               aspect='auto', cmap='plasma')
    plt.colorbar(label='Magnitude of Coefficients')
    plt.xlabel('MJD')
    plt.ylabel(f'Scale (Resolution: {resolution})')
    plt.title(f'Wavelet Analysis at Resolution {resolution}')
    plt.show()



# Step 6: Cross-Correlation with Hellings-Downs Curve
theta = np.linspace(0, np.pi, len(all_MJD))
hellings_down_curve = 0.5 - 0.25 * np.cos(theta) + 0.5 * np.log(0.5 * (1 + np.cos(theta)))

# Normalize residuals
normalized_residuals = (all_timing_residuals - np.mean(all_timing_residuals)) / np.std(all_timing_residuals)


# Cut Timing Residuals for Wavelet Decomposition
time_window_size = 300  # Define the size of each time window
num_windows = len(all_MJD) // time_window_size

for i in range(num_windows):
    window_start = i * time_window_size
    window_end = window_start + time_window_size
    timing_window = all_timing_residuals[window_start:window_end]
    MJD_window = all_MJD[window_start:window_end]

    # Perform Wavelet Analysis on each window
    window_coefficients, _ = pywt.cwt(timing_window, scales, 'cmor')
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(window_coefficients), extent=[MJD_window[0], MJD_window[-1], scales[-1], scales[0]],
               aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude of Coefficients')
    plt.xlabel('MJD (Windowed)')
    plt.ylabel('Scale')
    plt.title(f'Wavelet Analysis for Time Window {i + 1}')
    plt.show()

    # Compare with Hellings-Downs curve for the window
    theta_window = np.linspace(0, np.pi, len(MJD_window))
    hellings_down_curve_window = 0.5 - 0.25 * np.cos(theta_window) + 0.5 * np.log(0.5 * (1 + np.cos(theta_window)))

    normalized_window_residuals = (timing_window - np.mean(timing_window)) / np.std(timing_window)
    correlation_window = correlate(normalized_window_residuals, hellings_down_curve_window, mode='full')
    lags_window = np.arange(-len(hellings_down_curve_window) + 1, len(normalized_window_residuals))

    plt.figure(figsize=(10, 6))
    plt.plot(lags_window, correlation_window, label='Cross-Correlation', color='b', linewidth=1.5)
    plt.plot(np.arange(len(hellings_down_curve_window)),
             hellings_down_curve_window * max(correlation_window) / max(hellings_down_curve_window),
             label='Hellings-Downs Curve (Scaled)', color='r', linestyle='--', linewidth=1.5)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Lags / Angular Separation (radians)')
    plt.ylabel('Amplitude (Normalized)')
    plt.title(f'Cross-Correlation with Hellings-Downs Curve for Window {i + 1}')
    plt.legend()
    plt.grid()
    plt.show()
# Cutting the timing residual plot into wavelets by varying resolution
wavelet_cut_resolutions = [0.25, 0.5, 1.0, 2.0]
for resolution in wavelet_cut_resolutions:
    adjusted_scales = scales * resolution
    cut_coefficients, _ = pywt.cwt(all_timing_residuals, adjusted_scales, 'cmor')
    
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(cut_coefficients), extent=[all_MJD[0], all_MJD[-1], adjusted_scales[-1], adjusted_scales[0]], 
               aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Coefficient Magnitude')
    plt.xlabel('MJD')
    plt.ylabel(f'Scale (Resolution: {resolution})')
    plt.title(f'Wavelet Cut at Resolution {resolution}')
    plt.show()

    # Check match with Hellings-Downs curve
    for scale_idx, scale in enumerate(adjusted_scales):
        current_wavelet = cut_coefficients[scale_idx, :]
        current_correlation = correlate(current_wavelet, hellings_down_curve, mode='full')
        
        plt.figure(figsize=(10, 4))
        plt.plot(current_correlation, label=f'Scale {scale:.2f} (Resolution: {resolution})', color='b')
        plt.title(f'Correlation at Scale {scale:.2f} (Resolution: {resolution})')
        plt.xlabel('Lags')
        plt.ylabel('Cross-Correlation')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.legend()
        plt.grid()
        plt.show()




# Code from /mnt/data/pcap1.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

# Directory containing PTA data files
data_dir = "/users/adityan/downloads/ptadata"
output_csv = "pta_timing_residuals.csv"

def load_pta_data(directory):
    """Load PTA timing residuals from all txt files in the given directory."""
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            pulsar_name = filename.split(".")[0]  # Extract pulsar name from filename
            
            # Load data (handle multiple columns by selecting only MJD and Residuals)
            raw_data = np.loadtxt(filepath)
            if raw_data.shape[1] >= 2:  # Ensure at least two columns exist
                mjd = raw_data[:, 0]  # First column is MJD
                residuals = raw_data[:, 1]  # Second column is residuals
            else:
                print(f"Skipping {filename}: Not enough columns.")
                continue
            
            # Remove outliers using Z-score
            filtered_indices = np.abs(zscore(residuals)) < 3
            cleaned_residuals = residuals[filtered_indices]
            cleaned_mjd = mjd[filtered_indices]
            
            # Append to dataset
            data.append((pulsar_name, cleaned_mjd, cleaned_residuals))
    return data

def butter_filter(signal, cutoff, fs, order=4, btype='low'):
    """Apply a Butterworth filter to remove high or low-frequency noise."""
    if fs <= 0:
        print("Warning: Invalid sampling frequency. Using default fs = 1.")
        fs = 1  # Default fallback value
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff <= 0 or normal_cutoff >= 1:
        print(f"Warning: Invalid cutoff frequency {cutoff}. Adjusting to safe range.")
        normal_cutoff = min(max(normal_cutoff, 0.01), 0.99)  # Keep within valid range
    
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, signal)

def remove_noise(mjd, residuals):
    """Mathematically remove white noise and red noise using polynomial fitting and filtering."""
    if len(mjd) < 10:  # Ensure enough data points for meaningful filtering
        return residuals
    
    fs = 1 / np.median(np.diff(mjd)) if len(mjd) > 1 else 1  # Ensure valid sampling frequency
    
    # Remove white noise using polynomial fitting
    poly_fit = np.polyfit(mjd, residuals, 3)
    residuals = residuals - np.polyval(poly_fit, mjd)
    
    # Apply Butterworth filters
    residuals = butter_filter(residuals, cutoff=0.01, fs=fs, order=4, btype='low')  # Remove high-frequency noise
    residuals = butter_filter(residuals, cutoff=0.0001, fs=fs, order=4, btype='high')  # Remove low-frequency drift
    
    return residuals

def align_data(data):
    """Interpolates all pulsar timing residuals to a common set of time points."""
    all_times = np.unique(np.concatenate([mjd for _, mjd, _ in data]))
    aligned_data = {}
    for pulsar, mjd, residuals in data:
        interp_func = interp1d(mjd, residuals, kind='linear', fill_value='extrapolate')
        aligned_data[pulsar] = interp_func(all_times)
    return all_times, np.array(list(aligned_data.values()))

def apply_pca(aligned_data):
    """Applies PCA to the aligned data matrix."""
    pca = PCA(n_components=2)  # Adjust components as needed
    principal_components = pca.fit_transform(aligned_data.T)  # Transpose for PCA input
    return principal_components, pca.explained_variance_ratio_

def save_to_csv(data, filename):
    """Save processed timing residuals to CSV."""
    df_list = []  # List to store DataFrames before concatenation
    for pulsar_name, mjd, residuals in data:
        temp_df = pd.DataFrame({"Pulsar": [pulsar_name] * len(mjd), "MJD": mjd, "Residuals": residuals})
        df_list.append(temp_df)
    
    if df_list:  # Ensure df_list is not empty before concatenation
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(filename, index=False)
        print(f"Timing residuals saved to {filename}")
    else:
        print("Warning: No data to save. Check input files.")

# Main execution
pta_data = load_pta_data(data_dir)
pta_data = [(name, mjd, remove_noise(mjd, residuals)) for name, mjd, residuals in pta_data]
time_grid, aligned_matrix = align_data(pta_data)
pca_results, variance_ratio = apply_pca(aligned_matrix)

# Save or visualize PCA results
print("PCA Results Shape:", pca_results.shape)
print("Explained Variance Ratio:", variance_ratio)

save_to_csv(pta_data, output_csv)

# Visualization of PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=time_grid, cmap='viridis', edgecolor='k')
plt.colorbar(label='MJD')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Pulsar Timing Residuals')
plt.show()


# Code from /mnt/data/asw4.py
import numpy as np
import scipy.signal as signal
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
import os

def power_law(f, A, gamma):
    return A * f**(-gamma)

def fit_power_law(f, Pxx):
    try:
        log_f = np.log(f)
        log_Pxx = np.log(Pxx)
        popt, _ = opt.curve_fit(lambda x, A, gamma: np.log(A) - gamma * x, log_f, log_Pxx, p0=[1e-10, 4], maxfev=5000)
        return np.exp(popt[0]), popt[1]  # Convert log(A) back to A
    except RuntimeError:
        print("Curve fitting failed, using default values.")
        return 1e-10, 4  # Default values if fitting fails

def process_pta_data(directory):
    results = {}
    plt.figure(figsize=(10, 6))
    
    all_frequencies = []
    all_psd = []
    pulsar_labels = []
    common_frequencies = np.linspace(0.05, 250, 300)  # Common frequency bins for interpolation
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        try:
            data = np.loadtxt(filepath)
            if data.shape[1] < 3:
                print(f"Skipping {filepath}: Insufficient data columns.")
                continue
            
            mjd = data[:, 0]
            dm = data[:, 1]
            
            print(f"{filename}: MJD time differences (first 5): {np.diff(mjd)[:5]}")
            
            time_seconds = (mjd - mjd[0]) * 86400.0
            
            if not np.all(np.diff(time_seconds) > 0):
                print(f"Warning: Time not strictly increasing in {filename}, sorting data.")
                sorted_indices = np.argsort(time_seconds)
                time_seconds = time_seconds[sorted_indices]
                dm = dm[sorted_indices]
            
            nperseg = max(16, len(dm) // 2)
            
            fs = 1 / np.median(np.diff(time_seconds))
            f, Pxx = signal.welch(dm, fs=fs, nperseg=nperseg, scaling='density')
            
            f = np.abs(f)
            Pxx /= np.max(Pxx)
            
            f_nHz = f * 1e9
            
            print(f"{filename}: Full frequency range before filtering: {f_nHz.min():.2e} Hz to {f_nHz.max():.2e} Hz")
            
            valid_indices = (f_nHz >= 0.05) & (f_nHz <= 250)
            f_nHz = f_nHz[valid_indices]
            Pxx = Pxx[valid_indices]
            
            if len(f_nHz) == 0:
                print(f"Skipping {filename}: No valid frequencies in the desired range.")
                continue
            
            print(f"{filename}: Sample PSD values: {Pxx[:5]}")
            
            A, gamma = fit_power_law(f_nHz, Pxx)
            results[filename] = {"A": A, "gamma": gamma}
            print(f"Processed {filename}: A = {A:.2e}, gamma = {gamma:.2f}")
            
            from scipy.interpolate import interp1d
            interpolator = interp1d(f_nHz, Pxx, kind='linear', fill_value='extrapolate')
            Pxx_interpolated = interpolator(common_frequencies)
            
            plt.plot(common_frequencies, Pxx_interpolated, label=f"{filename} (A={A:.2e}, γ={gamma:.2f})", alpha=0.8)
            
            all_psd.append(Pxx_interpolated)
            pulsar_labels.append(filename)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    plt.xlabel("Frequency (nHz)")
    plt.ylabel("Power Spectral Density")
    plt.title("GWB Amplitude Across Pulsars with Extracted Parameters")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    
    # 1. Mean Trend Plot
    plt.figure(figsize=(10, 6))
    mean_psd = np.mean(np.array(all_psd), axis=0)
    plt.plot(common_frequencies, mean_psd, color='black', lw=2, label='Mean PSD')
    plt.fill_between(common_frequencies, mean_psd * 0.8, mean_psd * 1.2, color='gray', alpha=0.3)
    plt.xlabel("Frequency (nHz)")
    plt.ylabel("Mean Power Spectral Density")
    plt.title("Average GWB Signature Across Pulsars")
    plt.legend()
    plt.show()
    
    # 2. Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(np.array(all_psd), cmap='magma', yticklabels=pulsar_labels, xticklabels=False)
    plt.xlabel("Frequency Bins")
    plt.ylabel("Pulsars")
    plt.title("PSD Heatmap Across Pulsars")
    plt.show()
    
    return results

pta_directory = "/users/adityan/downloads/ptadata"
results = process_pta_data(pta_directory)
print("Extracted Amplitude and Spectral Index:")
for pulsar, values in results.items():
    print(f"{pulsar}: A = {values['A']:.3e}, gamma = {values['gamma']:.2f}")

