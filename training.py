import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


################### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


################# Function for training the model
def train_model():
    training_data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    X = training_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = training_data['exited']

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=100,
        n_jobs=None,
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    model.fit(X, y)

    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as f:
        pickle.dump(model, f)

    return model


if __name__ == '__main__':
    train_model()
