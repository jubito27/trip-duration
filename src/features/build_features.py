import pathlib
import pandas as pd
import numpy as np
from feature_definitions import test_feature_build
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df , test_size , seed):
    train, test = train_test_split(df , test_size=test_size , random_state=seed)
    return train  , test

def save_data(train , test , output_path):
    pathlib.Path(output_path).mkdir(parents=True , exist_ok=True)
    train.to_csv(output_path + '/train.csv' , index=False)
    test.to_csv(output_path + '/test.csv' , index=False)




if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    train_path = home_dir.as_posix() + '/data/raw/train.csv'
    test_path = home_dir.as_posix() + '/data/raw/test.csv'
    print(curr_dir , home_dir , train_path , test_path)
    train = pd.read_csv(train_path, nrows=10)
    test = pd.read_csv(test_path , nrows=10)
    # print(test_feature_build(train , test))
    output_path = home_dir.as_posix() + 'data/processed'
