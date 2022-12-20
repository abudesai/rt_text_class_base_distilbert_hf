# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

import io
import pandas as pd, numpy as np
import json
import flask
import traceback
import sys
import os, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import algorithm.utils as utils
from algorithm.model_server import ModelServer
from algorithm.model import classifier as model

prefix = "/opt/ml_vol/"
data_schema_path = os.path.join(prefix, "inputs", "data_config")
model_path = os.path.join(prefix, "model", "artifacts")
failure_path = os.path.join(prefix, "outputs", "errors", "serve_failure")


# get data schema - its needed to set the prediction field name
# and to filter df to only return the id and pred columns
data_schema = utils.get_data_schema(data_schema_path)


# initialize your model here before the app can handle requests
model_server = ModelServer(model_path=model_path, data_schema=data_schema)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy."""
    status = 200
    response = f"Hello - I am {model.MODEL_NAME} model and I am at your service!"
    print(response)
    return flask.Response(response=response, status=status, mimetype="application/json")


@app.route("/infer", methods=["POST"])
def infer():
    """Do an inference on a single batch of data. In this sample server, we take data as a JSON object, convert
    it to a pandas data frame for internal use and then convert the predictions back to JSON.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "application/json":
        req_data_dict = json.loads(flask.request.data.decode("utf-8"))
        data = pd.DataFrame.from_records(req_data_dict["instances"])
    else:
        return flask.Response(
            response="This endpoint only supports application/json data",
            status=415,
            mimetype="text/plain",
        )

    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    try:
        predictions_df = model_server.predict_proba(data)
        predictions_df.columns = [str(c) for c in predictions_df.columns]
        class_names = predictions_df.columns[1:]

        predictions_df["__label"] = pd.DataFrame(
            predictions_df[class_names], columns=class_names
        ).idxmax(axis=1)

        # convert to the json response specification
        id_field_name = model_server.id_field_name
        predictions_response = []
        for rec in predictions_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[id_field_name] = rec[id_field_name]
            pred_obj["label"] = rec["__label"]
            pred_obj["probabilities"] = {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [id_field_name, "__label"]
            }
            predictions_response.append(pred_obj)

        return flask.Response(
            response=json.dumps({"predictions": predictions_response}),
            status=200,
            mimetype="application/json",
        )

    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        error_msg = "Exception during inference: " + str(err) + "\n" + trc
        with open(failure_path, "w") as s:
            s.write(error_msg)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during inference: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        response = json.dumps({"error": str(err)})
        return flask.Response(
            response=response,
            status=400,
            mimetype="text/plain",
        )


@app.route("/infer_file", methods=["POST"])
def infer_file():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s)
    else:
        return flask.Response(
            response="This predictor only supports CSV data",
            status=415,
            mimetype="text/plain",
        )

    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    try:
        predictions = model_server.predict_proba(data)
        # Convert from dataframe to CSV
        out = io.StringIO()
        predictions.to_csv(out, index=False)
        result = out.getvalue()

        return flask.Response(response=result, status=200, mimetype="text/csv")

    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        error_msg = "Exception during inference: " + str(err) + "\n" + trc
        with open(failure_path, "w") as s:
            s.write(error_msg)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during inference: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        response = json.dumps({"error": str(err)})
        return flask.Response(
            response=response,
            status=400,
            mimetype="text/plain",
        )
