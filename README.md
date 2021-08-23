# Project: Operationalizing Machine Learning

## Automated ML Experiment and Endpoint

I've started by creating an AutoML run using the *Bank Marketing* dataset provided.

The dataset was first registered into the azure ml workspace:

![screenscreenshots\dataset.PNG](screenshots\dataset.PNG)

then an *automl* run was created and run using the code provided into `lancia\set_up_part1.ipynb`

```{python}
from azureml.train.automl import AutoMLConfig

automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
}
automl_config = AutoMLConfig(compute_target=cpu_cluster,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="y",   
                             path = "./test",
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings)
                            
from azureml.core import Experiment


experiment_name = 'AZMLE-project2-step2'
experiment = Experiment(ws, experiment_name)

from azureml.widgets import RunDetails
from azureml.core.run import Run

automl = experiment.submit(automl_config)
RunDetails(automl).show()
automl.wait_for_completion(show_output=True)

```

resulting in an experiment run called `AZMLE-project2-step2` logged here:

![screenshots\experiment_run.PNG](screenshots\experiment_run.PNG)

the best model achieved is a voting ensamble as in the previous project.

![model](screenshots\best_model.PNG)

The model was deploy first without any insight enabled:

![no-ins](screenshots\deployed_no_insight.PNG)

then using the code in `lancia\enable_insight.py` insight were enabled

```{python}
from azureml.core import Workspace
from azureml.core.webservice import Webservice

# Requires the config to be downloaded first to the current working directory
ws = Workspace.from_config()

# Set with the deployment name
name = "bank-marketing"




# Load existing web service
service = Webservice(name=name, workspace=ws)

# Enable Application Insights
service.update(enable_app_insights=True)
```

obtaining

![insights](screenshots\deployed_w_insight.PNG)

**Important** the code requires to authenticate to the correct workspace thus a `config.json` file is needed in the same or parents directory.

With a endopoint up and running we can use the file `log.py` to retrieve the logs:

![log](screenshots\log.PNG)

When the endpoint is up and running we can use the swagger.json file with the Swagger tool to get info and documentation about the exposed REST API.

We created a webserver to expose the swagger.json file from the same domain as the swagger service and obtained:

![swagger](screenshots\swagger1.PNG)
![swagger](screenshots\swagger2.PNG)

I've modified the `endopoint.py` adding the authentication parameters required for the API **and** by rearranging the payload of the request to match the signature from the swagger doc.

Without this change the system will produce an unhandled exception.

```{python}
import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://469741b0-acef-4a4e-a46d-845c72d93509.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'EuNmM1Ccr2TniZYvreexh75h9BzEA0S4'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 17,
            "job": "blue-collar",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "yes",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "duration": 971,
            "campaign": 1,
            "pdays": 999,
            "previous": 1,
            "poutcome": "failure",
            "emp.var.rate": -1.8,
            "cons.price.idx": 92.893,
            "cons.conf.idx": -46.2,
            "euribor3m": 1.299,
            "nr.employed": 5099.1
          }
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```

![endpont](screenshots\endpoint.PNG)

This script run produced a `data.json` and proved working. So i can test the endpoint performances using apache bench.

![bench](screenshots\bench1.PNG)
![bench](screenshots\bench.PNG)


## Pipeline

Using the provied notebook we created a pipeline

![pipe_crated](screenshots\pipepline_created.PNG)

and we exposed that pipelien with an endpoint

![pipe_crated](screenshots\pipe_endpoint.PNG)

The pipeline is composed of two steps:

* the dataset
* the automl module

![pipe_strucure](screenshots\pipe_overview.PNG)

in the same screenshot above we can see the pipeline overview showing hte REST endopint

If we call this endpoint from a jupyter notebook we can see in the widged the status of the run and the structure of the pipeline

![pipe_vidjet](screenshots\widget.PNG)

and the submitted runs in the azure ml studio

![pipe_runs](screenshots\runs.PNG)

## Screencast link

https://youtu.be/j2u5UMYq2BM