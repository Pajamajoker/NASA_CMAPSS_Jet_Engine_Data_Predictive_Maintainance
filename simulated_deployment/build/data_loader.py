# data_loader.py
import pandas as pd

COLUMN_NAMES = (
    ['unit_number', 'time_in_cycles',
     'operational_setting_1', 'operational_setting_2', 'operational_setting_3']
    + [f'sensor_measurement_{i}' for i in range(1,22)]
)

def load_data(train_path, test_path, rul_path):
    train = pd.read_csv(train_path, sep=' ', header=None).drop(columns=[26,27])
    train.columns = COLUMN_NAMES
    test  = pd.read_csv(test_path,  sep=' ', header=None).drop(columns=[26,27])
    test.columns  = COLUMN_NAMES
    rul_df = pd.read_csv(rul_path, header=None)
    return train, test, rul_df

def compute_rul(train_df, test_df, rul_df):
    # train RUL
    max_cycle = train_df.groupby('unit_number')['time_in_cycles'].max()
    train_df['RUL'] = train_df.apply(
        lambda r: min(max_cycle[r.unit_number] - r.time_in_cycles, 165),
        axis=1
    )
    # test RUL: last cycle + provided RUL
    last = test_df.groupby('unit_number').last().reset_index()
    last['RUL'] = rul_df.values
    return train_df, last
