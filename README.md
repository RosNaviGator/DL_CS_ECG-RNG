# This branch is completely random

It was during work in progress, it's here only to keep the tests with other adaptive learning dictionaries like the one from sklearn

It's a mess...



Needs version of python < 3.12 for sparseland (3.8)

```sh
deactivate
rm -r .adaptLearnVenv
python -m venv .adaptLearnVenv
source .adaptLearnVenv/bin/activate
pip install numpy scipy
pip install pandas PyWavelets
pip install scikit-learn
```




## Interpreting `n_samples` and `n_features` for ECG Data

When working with an ECG signal (or any other time series data), the concepts of `n_samples` and `n_features` are interpreted differently compared to a standard tabular dataset. Let's clarify these terms in the context of ECG data:

### ECG Data Representation

- **ECG Signal as Time Series**: An ECG (electrocardiogram) signal is a time series where each data point represents a voltage measurement (amplitude) at a specific time instant. The signal is typically sampled at a fixed rate (e.g., 250 Hz, 500 Hz, etc.), meaning there are several measurements per second.

- **Segments of ECG Signal**: For processing with machine learning models, we often segment the ECG signal into fixed-length windows. Each window can be treated as a sample.

### Definition of `n_samples` and `n_features` for ECG Signals

- **`n_samples`**: This would represent the number of ECG signal segments (windows) you have. If you divide the entire ECG signal into multiple overlapping or non-overlapping segments of fixed length, each segment would be considered a sample.

- **`n_features`**: This would be the number of time points (samples) in each segment. Essentially, it's the length of each ECG segment (window) in terms of the number of time steps or measurements.
