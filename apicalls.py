import json
import os

import requests


with open('config.json', 'r') as f:
    config = json.load(f)

# Specify a URL that resolves to your workspace
URL = 'http://127.0.0.1:8000'
output_model_path = os.path.join(config['output_model_path'])


def call_apis():
    """Call all four API endpoints, combine the responses, and persist them."""
    # Call each API endpoint and store the responses
    response1 = requests.post(f'{URL}/prediction', json={'filepath': 'testdata/testdata.csv'})
    response2 = requests.get(f'{URL}/scoring')
    response3 = requests.get(f'{URL}/summarystats')
    response4 = requests.get(f'{URL}/diagnostics')

    # Combine all API responses into a single record
    responses = {
        'prediction': response1.json(),
        'scoring': response2.json(),
        'summarystats': response3.json(),
        'diagnostics': response4.json(),
    }

    # Write the combined responses to output_model_path
    os.makedirs(output_model_path, exist_ok=True)
    output_file = os.path.join(output_model_path, 'apireturns.txt')
    with open(output_file, 'w') as f:
        f.write(json.dumps(responses, indent=2))

    return responses


if __name__ == '__main__':
    try:
        call_apis()
    except requests.exceptions.RequestException as exc:
        raise SystemExit(
            f'Failed to reach the API at {URL}. Is the Flask app running? Details: {exc}'
        )
