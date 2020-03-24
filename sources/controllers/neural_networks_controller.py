import json

from flask import Blueprint
from bson import json_util
from bson.objectid import ObjectId

from sources.services.neural_networks_service import NeuralNetworksService

neural_networks = Blueprint(
    'neural_networks', __name__, url_prefix='/api/v1/neural-networks')


@neural_networks.route('/', methods=['GET'])
def get_models():
    results = NeuralNetworksService().get_neural_networks()
    json_results = []
    for result in results:
        json_results.append(result)
    return json.dumps(json_results, default=json_util.default)


@neural_networks.route('/<id>', methods=['GET'])
def get_model(id):
    result = NeuralNetworksService().get_neural_network(id)
    return json.dumps(result, default=json_util.default)


@neural_networks.route('/status', methods=['GET'])
def get_total_models():
    result = NeuralNetworksService().get_neural_networks_status()
    return json.dumps(result, default=json_util.default)
