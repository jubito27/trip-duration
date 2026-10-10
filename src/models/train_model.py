import pathlib
import sys 
import yaml
import joblib

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


def train_model(featues , target , max_depth , n_estimators , seed):
    model = RandomForestRegressor(max_depth=max_depth , n_estimators=n_estimators , random_state=seed)
    model.fit(featues , target)
    return model

def save_model(model , output_path):
    joblib.dump(model , output_path + '/model.joblib')

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    param_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(param_file))['train_model']

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(exist_ok=True , parents=True)

    TARGET= 'trip_duration'
    features= pd.read_csv(data_path + '/train.csv')
    X = features.drop(TARGET)
    y = features[TARGET]
    trained_model = train_model(X , y , params['max_depth'] , params['n_estimators'] , params['seed'])
    save_model(train_model , output_path)


if __name__ == '__main__':
    main()
    