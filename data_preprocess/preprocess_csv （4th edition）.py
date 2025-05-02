"""
5G Data Preprocessing Script
Group Members: A.Lin, B.Chen, Y.Mao
"""

import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

csv_files = glob.glob('*.csv')
print(f"\nFound {len(csv_files)} CSV files:")
for file in csv_files:
    print(f" - {file}")

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

full_data = pd.concat(dfs, ignore_index=True)
print(f"Total records after merging: {len(full_data)}")

full_data.replace('-', np.nan, inplace=True)
full_data.dropna(inplace=True)
print(f"Total records after cleaning: {len(full_data)}")

df_5g = full_data[full_data['NetworkMode'] == '5G']
print(f"5G data samples: {len(df_5g)}")

df_5g.loc[:, 'DL_diff'] = df_5g['DL_bitrate'].diff().fillna(0)
df_5g.loc[:, 'UL_diff'] = df_5g['UL_bitrate'].diff().fillna(0)

resource_features = df_5g[['DL_bitrate', 'UL_bitrate', 'DL_diff', 'UL_diff']].values
channel_features = df_5g[['SNR', 'RSRP', 'RSRQ']].values

res_scaler = StandardScaler()
chan_scaler = StandardScaler()
resource_features = res_scaler.fit_transform(resource_features)
channel_features = chan_scaler.fit_transform(channel_features)

time_steps = 10
res_X, res_y = [], []
chan_X, chan_y = [], []

for i in range(len(df_5g) - time_steps):
    res_seq = resource_features[i:i+time_steps]
    chan_seq = channel_features[i:i+time_steps]

    final_dl = resource_features[i+time_steps-1, 0]
    total_dl_ul = np.sum(resource_features[i+time_steps-1, :2])
    res_label = final_dl / total_dl_ul if total_dl_ul != 0 else 0.0

    avg_snr = chan_seq[:, 0].mean()
    if avg_snr > 20:
        chan_label = [1, 0, 0, 0, 0, 0]
    elif avg_snr > 10:
        chan_label = [0, 1, 0, 0, 0, 0]
    elif avg_snr > 0:
        chan_label = [0, 0, 1, 0, 0, 0]
    elif avg_snr > -10:
        chan_label = [0, 0, 0, 1, 0, 0]
    elif avg_snr > -20:
        chan_label = [0, 0, 0, 0, 1, 0]
    else:
        chan_label = [0, 0, 0, 0, 0, 1]

    res_X.append(res_seq)
    res_y.append(res_label)
    chan_X.append(chan_seq)
    chan_y.append(chan_label)

res_X = np.array(res_X)
res_y = np.array(res_y)
chan_X = np.array(chan_X)
chan_y = np.array(chan_y)

print(f"Generated {len(res_X)} samples with sequence length {time_steps}")

res_X_train, res_X_test, res_y_train, res_y_test = train_test_split(
    res_X, res_y, test_size=0.3, random_state=42)

chan_X_train, chan_X_test, chan_y_train, chan_y_test = train_test_split(
    chan_X, chan_y, test_size=0.3, random_state=42)

print("Completed train-test split.")

np.savez('processed_data.npz',
         res_X_train=res_X_train, res_y_train=res_y_train,
         res_X_test=res_X_test, res_y_test=res_y_test,
         chan_X_train=chan_X_train, chan_y_train=chan_y_train,
         chan_X_test=chan_X_test, chan_y_test=chan_y_test)

print("Preprocessed dataset saved as 'processed_data.npz'.")
