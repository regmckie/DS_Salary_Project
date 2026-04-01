import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle


def load_models():
    """
    Loads trained model from disk
    """
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model


app = Flask(__name__) # Create and initialize Flask application
@app.route('/predict', methods=['GET'])  # Creates the HTTP endpoint
def predict():
    """
    Makes and outputs predictions based on the data sent from HTTP request
    """
    # Get our input data sent from the HTTP request
    request_json = request.get_json()
    x = request_json['input']
    x_in = np.array(x).reshape(1, -1)

    # Load model
    model = load_models()

    # Makes and outputs prediction by sending the response back to the client
    prediction = model.predict(x_in)[0]
    response = json.dumps({'response': prediction})
    return response, 200


if __name__ == '__main__':
    application.run(debug=True)