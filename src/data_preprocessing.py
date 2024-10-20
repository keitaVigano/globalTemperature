import pandas as pd
from sktime.split import TemporalTrainTestSplitter

def load_and_prepare_data(file_path):
    """Load and drop unnecessary columns"""
    df = pd.read_csv(file_path, sep=",")
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index("dt", inplace=True)
    df.drop(['LandMaxTemperature', 'LandMaxTemperatureUncertainty',
             'LandMinTemperature', 'LandMinTemperatureUncertainty',
             'LandAndOceanAverageTemperature',
             'LandAndOceanAverageTemperatureUncertainty'], axis=1, inplace=True)
    return df

def split_data(df, test_size=0.2):
    """split and also export data"""
    splitter = TemporalTrainTestSplitter(test_size=test_size)
    train_indices, test_indices = next(splitter.split(df["LandAverageTemperature"]))
    df_train = df.iloc[train_indices]
    df_test = df.iloc[test_indices]
    df_train.to_csv("../data/cleaned/GlobalTemperaturesCleanedTrain.csv", index = True, index_label = "Time")
    df_test.to_csv("../data/cleaned/GlobalTemperaturesCleanedTest.csv", index = True, index_label = "Time")
    return df_train, df_test
