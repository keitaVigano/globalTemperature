import pandas as pd

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, sep=",")
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index("dt", inplace=True)
    df.drop(['LandMaxTemperature', 'LandMaxTemperatureUncertainty',
             'LandMinTemperature', 'LandMinTemperatureUncertainty',
             'LandAndOceanAverageTemperature',
             'LandAndOceanAverageTemperatureUncertainty'], axis=1, inplace=True)
    return df