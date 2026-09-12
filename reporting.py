import json
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless plotting

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

import diagnostics


############### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
model_output_path = os.path.join(config['output_model_path'])


############## Function for reporting
def score_model():
    # Calculate a confusion matrix using the test data and the deployed model
    # and write the confusion matrix to the workspace
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_true = test_data['exited']
    y_pred = diagnostics.model_predictions(test_data)

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    os.makedirs(model_output_path, exist_ok=True)
    output_path = os.path.join(model_output_path, 'confusionmatrix.png')

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


if __name__ == '__main__':
    score_model()
