# Dynamic Risk Assessment System

An MLOps pipeline that predicts client attrition risk and keeps the deployed model healthy over time. It ingests data, trains and scores a logistic regression model, deploys it, runs diagnostics and reporting, exposes results through a Flask API, and automates monitoring with drift-based retraining on a schedule.

## Pipeline Stages
- **Ingestion** (`ingestion.py`) — merges every CSV in `input_folder_path`, drops duplicates, and writes `finaldata.csv` + `ingestedfiles.txt`.
- **Training** (`training.py`) — fits a scikit-learn `LogisticRegression` on the numeric features and saves `trainedmodel.pkl`.
- **Scoring** (`scoring.py`) — computes the F1 score on the test data and writes `latestscore.txt`.
- **Deployment** (`deployment.py`) — copies `trainedmodel.pkl`, `latestscore.txt`, and `ingestedfiles.txt` into `prod_deployment_path`.
- **Diagnostics** (`diagnostics.py`) — summary statistics, missing-data percentages, ingestion/training timings, dependency versions, and predictions.
- **Reporting** (`reporting.py`) — generates the confusion matrix plot `confusionmatrix.png`.
- **API** (`app.py`) — Flask endpoints `/prediction`, `/scoring`, `/summarystats`, `/diagnostics`.
- **API aggregation** (`apicalls.py`) — calls all four endpoints and combines results into `apireturns.txt`.
- **Automation** (`fullprocess.py` + `cronjob.txt`) — checks for new data, detects model drift, retrains/redeploys when needed, and refreshes reports.

Paths are driven by `config.json` (`input_folder_path`, `output_folder_path`, `test_data_path`, `output_model_path`, `prod_deployment_path`).

## How to Run

### 1. Set up the environment
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline stages (in order)
```bash
python ingestion.py     # -> ingesteddata/finaldata.csv, ingestedfiles.txt
python training.py      # -> models/trainedmodel.pkl
python scoring.py       # -> models/latestscore.txt
python deployment.py    # -> production_deployment/{trainedmodel.pkl, latestscore.txt, ingestedfiles.txt}
python diagnostics.py   # prints diagnostics
python reporting.py     # -> models/confusionmatrix.png
```

### 3. Use the API
```bash
# Terminal 1: start the server (listens on port 8000)
python app.py

# Terminal 2: call all endpoints and aggregate the results
python apicalls.py      # -> models/apireturns.txt
```

### 4. Run the full automated process
```bash
python fullprocess.py   # checks for new data + drift; retrains/redeploys only if both are present
```

### 5. Schedule it (optional)
`cronjob.txt` holds a one-line entry that runs `fullprocess.py` every 10 minutes. It is not installed automatically; load it into your user crontab if you want it active:
```bash
crontab cronjob.txt     # note: this replaces your current crontab
```

---

# Step 1: Data Ingestion

## Purpose
- Build a flexible ingestion process for changing input datasets
- Combine multiple CSV files into one master dataset

## Starter Files
- `ingestion.py`
- `config.json`
- datasets in `/practicedata/`

## Main Tasks
- Read all CSV files from the folder set in `config.json`
- Do not hardcode file names
- Combine all files into one pandas DataFrame
- Remove duplicate rows
- Save the final dataset as `finaldata.csv`
- Save the list of ingested filenames as `ingestedfiles.txt`

## Output Locations
- Save outputs to the folder in `output_folder_path`
- Starter config uses `/ingesteddata/`

## Important Config Settings
- `input_folder_path`: where input data is read from
- `output_folder_path`: where ingestion outputs are saved
- `test_data_path`: location of test data
- `output_model_path`: where models and scores are stored
- `prod_deployment_path`: where production models are stored

## Project Note
- Early setup uses `practicedata` and `practicemodels`
- Final project setup should use `sourcedata` and `models`

# Step 2: Training, Scoring, and Deploying

## Purpose
- Train a basic ML model that can be monitored and updated later
- Focus on having a working predictive model, not perfect accuracy

## Required Scripts
- `training.py`
- `scoring.py`
- `deployment.py`

## Training
- Read `finaldata.csv` from the folder in `output_folder_path`
- Train the provided logistic regression model with scikit-learn
- Save the trained model as `trainedmodel.pkl` in `output_model_path`

## Scoring
- Read test data from `test_data_path`
- Load `trainedmodel.pkl` from `output_model_path`
- Calculate the model’s F1 score on the test dataset
- Save the score to `latestscore.txt` in `output_model_path`

## Deployment
- Copy these files into the production deployment directory in `prod_deployment_path`:
  - `trainedmodel.pkl`
  - `latestscore.txt`
  - `ingestedfiles.txt`

## Files to Save for Submission
- completed `training.py`
- completed `scoring.py`
- `trainedmodel.pkl`
- `latestscore.txt`
- completed `deployment.py`

# Step 3: Model and Data Diagnostics

## Purpose
- Check the health of the model and dataset
- Detect issues in data quality, model behavior, timing, and dependencies

## Starter Files
- `diagnostics.py`
- `config.json`

## Required Functions
- Calculate summary statistics for each numeric column:
  - mean
  - median
  - mode
- Calculate the percentage of NA values in each column
- Measure execution time for:
  - data ingestion
  - model training
- Check whether module dependencies are up to date
- Generate predictions using the current deployed model

## Data Sources
- Use the dataset stored in the folder from `output_folder_path`
- Load the deployed model from `prod_deployment_path`

## Expected Outputs
- Return summary statistics as a Python list
- Return missing-data percentages as a list
- Return model predictions for each row in an input DataFrame
- Produce a dependency table showing:
  - module name
  - current installed version
  - latest available version

## Key Notes
- Only compute statistics for numeric columns
- Missing data means NA values
- Predictions output length should match the number of input rows
- Use `pip` to check dependency versions
- No dependency updates are required, only reporting
# Step 4: Model Reporting

## Purpose
- Create reports about model performance and diagnostics
- Make results accessible through API endpoints
- Support automated reporting for stakeholders and monitoring

## Starter Files
- `reporting.py`
- `app.py`
- `apicalls.py`

## Reporting Task
- Use the prediction function from `diagnostics.py`
- Run predictions on the test dataset from `test_data_path`
- Compare predicted values with actual values
- Generate a confusion matrix plot
- Save the plot as `confusionmatrix.png`
- Store the output in the folder from `output_model_path`

## API Tasks
Create API endpoints in `app.py` that return HTTP 200:
- `/prediction` → returns model predictions for an input dataset path
- `/scoring` → returns the model score from `scoring.py`
- `/summarystats` → returns summary statistics from `diagnostics.py`
- `/diagnostics` → returns:
  - timing results
  - missing-data results
  - dependency check results

## API Calls Script
- Use `apicalls.py` to call all API endpoints
- Use `/testdata/testdata.csv` for the prediction endpoint
- Combine all endpoint outputs into one file
- Save the combined output as `apireturns.txt`
- Store the file in the folder from `output_model_path`

## Files to Save for Submission
- completed `reporting.py`
- completed `app.py`
- `confusionmatrix.png


# Step 5: Process Automation

## Goal
Create a script that runs the complete model scoring and monitoring process, then configure a cron job to run it every 10 minutes.

## Files to Save
- `fullprocess.py`
- `cronjob.txt`
- `confusionmatrix2.png`
- `apireturns2.txt`

## Required Cron Setup
- Configure a cron job that runs `fullprocess.py` once every 10 minutes
- Save the one-line cron job in `cronjob.txt`

## What `fullprocess.py` Must Do

### 1. Check for New Data
- Read `ingestedfiles.txt` from the production deployment directory
- Check the folder in `input_folder_path`
- Identify files not already listed in `ingestedfiles.txt`
- If new files exist, run `ingestion.py`
- If no new files exist, stop the process

### 2. Check for Model Drift
- Read the deployed model score from `latestscore.txt`
- Use the deployed `trainedmodel.pkl` and newly ingested data to make predictions
- Run `scoring.py` on the new data
- Compare the new score with the deployed model score
- If the new score is lower, treat that as model drift

### 3. Retrain and Redeploy
- Run `training.py` using the latest ingested data
- Run `deployment.py` to deploy the retrained model and updated records
- Only do this if:
  - new data exists, and
  - model drift is detected

### 4. Run Diagnostics and Reporting
- Run `apicalls.py`
- Run `reporting.py`
- Save the updated confusion matrix as `confusionmatrix2.png`
- Save the API output as `apireturns2.txt`

## Config Updates
Before completing this step, update `config.json` so that:
- `input_folder_path` points to `/sourcedata/`
- `output_model_path` points to `/models/` instead of `/practicemodels/`

## Process Flow Summary
- Check for new data
- Stop if there is no new data
- Ingest new data if present
- Check for model drift
- Stop if no drift is detected
- Retrain and redeploy if drift is detected
- Run diagnostics and reporting on the latest deployed model
