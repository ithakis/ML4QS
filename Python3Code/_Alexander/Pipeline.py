from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.signal import butter, filtfilt
from pykalman import KalmanFilter
import numpy as np

class Interpolator(BaseEstimator, TransformerMixin):
    def __init__(self, Hz):
        self.Hz = Hz

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Interpolate data at the specified Hz rate
        time_interval = 1 / self.Hz  # Convert Hz to time interval in seconds
        interpData = X.resample(f'{time_interval}S').interpolate()
        return interpData



class LowPassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, cutoff, fs, order=5):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        
        def apply_filter(data):
            return filtfilt(b, a, data)

        X_filtered = X.copy()
        X_filtered.iloc[:, 1:-1] = X_filtered.iloc[:, 1:-1].apply(apply_filter, axis=0)
        return X_filtered


class KalmanFilterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        measures = X.drop(columns=["Time (ms)", "Label"])
        initialMeans = np.zeros(measures.shape[1])
        transMatrix = np.eye(measures.shape[1])
        obsMatrix = np.eye(measures.shape[1])
        kf = KalmanFilter(
            transition_matrices=transMatrix,
            observation_matrices=obsMatrix,
            initial_state_mean=initialMeans
        )
        maskedMeasures = np.ma.masked_invalid(measures)
        kf = kf.em(maskedMeasures, n_iter=self.n_iter)
        filteredMeans, _ = kf.filter(maskedMeasures)
        X_filtered = X.copy()
        X_filtered.iloc[:, 1:-1] = filteredMeans
        return X_filtered

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

Hz = 50  # Assuming a sample rate of 50 Hz, replace with your actual value
pipeline = Pipeline([
    ('Interpolate', InterpolateAndNormalize(time_freq='T', Hz=Hz)),  # 'T' is for minute frequency, adjust as needed
    ('standard_scaler', StandardScaler()),
    ('low_pass_filter', LowPassFilter(cutoff=0.5, fs=Hz)),
    ('kalman_filter', KalmanFilterTransformer(n_iter=5)),
    ('outlier_detection', LocalOutlierFactor())
])

# Example usage
if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    pipeline.fit(data)
    transformed_data = pipeline.transform(data)



# data should be a pandas DataFrame with your data
# pipeline.fit(data)
# transformed_data = pipeline.transform(data)
