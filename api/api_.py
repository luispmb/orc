# app.py - a minimal flask api using flask_restful
import os
from flask import Flask, request, jsonify
#from flask_restful import Resource, Api
from datetime import datetime
from dotenv import load_dotenv
load_dotenv('./.env')


app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train():
  try:
    app.logger.info('treinando')
    data = request.get_json()
    print('data', data)
    acc = trainModel(data)
    return jsonify({'result': acc}), 200
  except Exception as e:
    return jsonify({"result": { "error": e.message }}), 500


@app.route('/train', methods=['POST'])
def train():
  try:
    app.logger.info('treinando')
    data = request.get_json()
    acc = trainModel(data)
    return jsonify({'result': acc}), 200
  except Exception as e:
    return jsonify({"result": { "error": e.message }}), 500

@app.route('/predict', methods=['POST'])
def predict():
  try:
    app.logger.info('treinando')
    data = request.get_json()
    res = runModel(data)
    return jsonify({'result': res}), 200
  except Exception as e:
    return jsonify({"result": { "error": e.message }}), 500



@app.route('/', methods=['GET'])
def test():
  return jsonify({'result': True}), 200

@app.route('/kube', methods=['POST'])
def testCluster():
  try:
    #runjob()
    return jsonify({'result': True}), 200
  except Exception as e:
    return jsonify({"result": { "error": e.message }}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port="8080")
