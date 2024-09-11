import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import wfdb
from scipy.signal import resample


def load_ptb_xl(path, sampling_rate=100):
    # PTB-XL dataset is loaded from a csv file, the ecg_id is used as the index
    path = "/home/deadbytes/Documents/ML/tiny-transformer/Task 2/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    # scp_codes column contains diagnostic information in string format, which is converted to a dictionary format using eval.
    Y.scp_codes = Y.scp_codes.apply(lambda x: eval(x))
    X = []
    for i in tqdm(Y.index, desc="Loading ECG Signals"):  # progress bar
        # For each record, the corresponding ECG signal data is read, file path for each signal is constructed using the filename_hr
        data, _ = wfdb.rdsamp(os.path.join(path, Y.loc[i].filename_hr))
        X.append(data)
    X = np.array(X)

    if sampling_rate != 500:
        # The resample function from scipy.signal is used to adjust the number of samples to match the new sampling rate.
        X = resample(X, int(X.shape[1]*sampling_rate/500), axis=1)

    # computes the frequency of each label
    label_counts = Y.scp.apply(pd.Series).sum()
    # Labels that appear in more than 5% of the ECGs are selected
    selected_labels = label_counts[label_counts > 0.05 * len(Y)].index.tolist()
    # binary label matrix y is created where each entry is 1 if the label is present in the ECG record, and 0 otherwise.
    y = np.array(Y.scp_codes.apply(
        lambda x: [1 if label in x else 0 for label in selected_labels]).tolist())
    return X, y, selected_labels
