import ast
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time

import pandas as pd
from sklearn.metrics import f1_score

import deployment
import reporting
import scoring
import training


with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

FEATURE_COLUMNS = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET_COLUMN = 'exited'

API_HOST = '127.0.0.1'
API_PORT = 8000


def read_ingested_files():
    """Read the record of files already ingested into production."""
    ingested_file_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    if not os.path.exists(ingested_file_path):
        return []

    with open(ingested_file_path, 'r') as f:
        content = f.read().strip()

    if not content:
        return []

    # ingestion.py records the filenames as a Python list literal, e.g. "['a.csv']".
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return [line.strip() for line in content.splitlines() if line.strip()]


def discover_source_files():
    """List every .csv currently present in the input folder (no hardcoded names)."""
    return sorted(
        file_name
        for file_name in os.listdir(input_folder_path)
        if file_name.endswith('.csv')
    )


def score_deployed_model_on_new_data():
    """Score the currently deployed model against the freshly ingested data.

    Mirrors scoring.py's F1 computation but uses the deployed model
    (prod_deployment_path/trainedmodel.pkl) and the new finaldata.csv.
    """
    new_data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    X = new_data[FEATURE_COLUMNS]
    y = new_data[TARGET_COLUMN]

    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    return f1_score(y, predictions)


def read_deployed_score():
    """Read the F1 score of the currently deployed model."""
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        return float(f.read().strip())


def _pids_on_port(port):
    """Return the set of PIDs bound to the given TCP port (macOS/Linux via lsof)."""
    try:
        result = subprocess.run(
            ['lsof', '-t', f'-i:{port}'],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return set()

    return {int(pid) for pid in result.stdout.split() if pid.strip().isdigit()}


def _kill_pids(pids):
    for pid in pids:
        try:
            os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            pass
    # Give processes a moment to exit, then force-kill any stragglers.
    time.sleep(1)
    for pid in pids:
        try:
            os.kill(pid, 9)  # SIGKILL
        except ProcessLookupError:
            pass


def _wait_for_port(host, port, timeout=30):
    """Block until the API server accepts connections or the timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def refresh_diagnostics_and_reporting():
    """Run apicalls.py and reporting.py, then save the '2' submission variants.

    app.py runs with debug=True, which spawns a reloader child process, so the
    whole set of PIDs bound to the API port is killed on teardown to avoid orphans.
    """
    server = subprocess.Popen([sys.executable, 'app.py'])
    try:
        if not _wait_for_port(API_HOST, API_PORT):
            raise RuntimeError(f'API server did not start on port {API_PORT}.')

        # Run the standard API-call aggregation (writes apireturns.txt).
        subprocess.run([sys.executable, 'apicalls.py'], check=True)
    finally:
        # Kill every PID on the API port (parent + debug reloader child), then
        # make sure the process we launched is fully reaped.
        _kill_pids(_pids_on_port(API_PORT))
        try:
            server.terminate()
            server.wait(timeout=10)
        except Exception:
            server.kill()

    # Run the standard reporting (writes confusionmatrix.png).
    reporting.score_model()

    # Produce the "2" variants required for this refresh without disturbing the
    # originals (which reporting.py / apicalls.py still generate when run standalone).
    os.makedirs(output_model_path, exist_ok=True)

    confusion_matrix_path = os.path.join(output_model_path, 'confusionmatrix.png')
    api_returns_path = os.path.join(output_model_path, 'apireturns.txt')

    if os.path.exists(confusion_matrix_path):
        shutil.copy2(
            confusion_matrix_path,
            os.path.join(output_model_path, 'confusionmatrix2.png'),
        )

    if os.path.exists(api_returns_path):
        shutil.copy2(
            api_returns_path,
            os.path.join(output_model_path, 'apireturns2.txt'),
        )


def main():
    ingested_files = read_ingested_files()
    source_files = discover_source_files()
    new_files = sorted(set(source_files) - set(ingested_files))

    if not new_files:
        print('No new data found. Ending process.')
        return

    print(f'New data found: {new_files}. Running ingestion.')
    subprocess.run([sys.executable, 'ingestion.py'], check=True)

    deployed_score = read_deployed_score()
    new_score = score_deployed_model_on_new_data()
    print(f'Deployed score: {deployed_score}. New-data score: {new_score}.')

    # Model drift = the deployed model scores lower on the new data.
    if new_score >= deployed_score:
        print('No model drift detected. Ending process.')
        return

    print('Model drift detected. Retraining and redeploying.')
    training.train_model()
    # Refresh latestscore.txt for the retrained model before promoting it.
    scoring.score_model()
    deployment.store_model_into_pickle()

    refresh_diagnostics_and_reporting()
    print('Pipeline completed: model retrained, redeployed, and reports refreshed.')


if __name__ == '__main__':
    main()
