import numpy as np
import pandas as pd
def get_full_dataframe(path):
    df = pd.read_csv(path)
    df['lesion_id'] = df['lesion_id'].apply(lambda x: 'Not Null' if pd.notnull(x) else 'Null')

    def fill_missing_with_distribution(series, distribution):
        missing_indices = series[series.isna()].index
        filled_values = np.random.choice(distribution.index, size=len(missing_indices), p=distribution.values)
        series.loc[missing_indices] = filled_values
        return series

    for category in ['sex', 'anatom_site_general']:
        dis = df[category].value_counts(normalize=True)
        df[category] = fill_missing_with_distribution(df[category], dis)
    

    df['iddx_2'] = df['iddx_2'].fillna(df['iddx_1'])
    df['iddx_3'] = df['iddx_3'].fillna(df['iddx_2'])
    df['iddx_4'] = df['iddx_4'].fillna(df['iddx_3'])
    df['iddx_5'] = df['iddx_5'].fillna(df['iddx_4'])


    return df


csv_file="./data/train-metadata.csv"

df = pd.read_csv(csv_file)
df_complete = get_full_dataframe(csv_file)