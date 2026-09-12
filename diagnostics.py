import json
import os
import pickle
import subprocess
import sys
import timeit

import pandas as pd


################## Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

FEATURE_COLUMNS = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET_COLUMN = 'exited'


################## Function to get model predictions
def model_predictions(test_data):
    # Read the deployed model and a dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(test_data[FEATURE_COLUMNS])
    return predictions.tolist()


################## Function to get summary statistics
def dataframe_summary():
    # Calculate summary statistics here
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    numeric_data = data.select_dtypes(include=['number'])

    means = numeric_data.mean().tolist()
    medians = numeric_data.median().tolist()
    modes = numeric_data.mode().iloc[0].tolist()

    return [means, medians, modes]


################## Function to get percent of missing data
def missing_data():
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    return (data.isna().mean() * 100).tolist()


##################Function to get timings
def execution_time():
    # Calculate timing of ingestion.py and training.py
    start_time = timeit.default_timer()
    subprocess.run([sys.executable, 'ingestion.py'], check=True)
    ingestion_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    subprocess.run([sys.executable, 'training.py'], check=True)
    training_time = timeit.default_timer() - start_time

    return [ingestion_time, training_time]


################## Function to check dependencies
def outdated_packages_list():
    # Build a table of each dependency's module name, installed version,
    # and latest available version WITHOUT updating anything.
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '==' not in line:
                continue
            package, version = line.split('==', 1)
            requirements.append((package, version))

    # Currently installed versions (pip list is read-only).
    installed = {}
    try:
        installed_result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            check=True,
            capture_output=True,
            text=True,
        )
        installed = {
            item['name'].lower(): item['version']
            for item in json.loads(installed_result.stdout or '[]')
        }
    except (subprocess.CalledProcessError, ValueError):
        installed = {}

    # Latest available versions for packages pip reports as outdated.
    outdated = {}
    try:
        outdated_result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
            check=True,
            capture_output=True,
            text=True,
        )
        outdated = {
            item['name'].lower(): item['latest_version']
            for item in json.loads(outdated_result.stdout or '[]')
        }
    except (subprocess.CalledProcessError, ValueError):
        outdated = {}

    rows = []
    for package, pinned_version in requirements:
        key = package.lower()
        current_version = installed.get(key, pinned_version)
        # If pip does not report it as outdated, the current version is latest.
        latest_version = outdated.get(key, current_version)
        rows.append({
            'module': package,
            'current_version': current_version,
            'latest_version': latest_version,
        })

    return pd.DataFrame(rows, columns=['module', 'current_version', 'latest_version'])


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    print(model_predictions(data))
    print(dataframe_summary())
    print(missing_data())
    print(execution_time())
    print(outdated_packages_list())
