import json

from flask import Blueprint, request
from bson import json_util
from bson.objectid import ObjectId

from sources.services.models_service import ModelsService

models = Blueprint('models', __name__, url_prefix='/api/v1/models')


@models.route('/', methods=['GET'])
def get_models():
    results = ModelsService().get_models()
    json_results = []
    for result in results:
        json_results.append(result)
    return json.dumps(json_results, default=json_util.default)


@models.route('/<id>', methods=['GET'])
def get_model(id):
    result = ModelsService().get_model(id)
    return json.dumps(result, default=json_util.default)


@models.route('/status', methods=['GET'])
def get_models_status():
    result = ModelsService().get_models_status()
    return json.dumps(result, default=json_util.default)


@models.route('/create', methods=['POST'])
def create_model():
    if request.headers['Content-Type'] == 'application/json':
        result = ModelsService().create_model(request.json)
        return json.dumps({'success': True, 'data': {'id': str(result.inserted_id)}}), 200, {'ContentType': 'application/json'}
    else:
        return json.dumps({'result': '415 Unsuported Media Type'})


@models.route('/update/<id>', methods=['PUT'])
def update_model(id):
    if request.headers['Content-Type'] == 'application/json':
        ModelsService().update_model(id, request.json)
        return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    else:
        return json.dumps({'result': '415 Unsuported Media Type'}, default=json_util.default)


@models.route('/test/<id>', methods=['POST'])
def test_model(id):
    if request.headers['Content-Type'] == 'application/json':
        ModelsService().test_model(id, request.json)
        return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    else:
        return json.dumps({'result': '415 Unsuported Media Type'}, default=json_util.default)


@models.route('/train/<id>', methods=['POST'])
def train_model(id):
    if request.headers['Content-Type'] == 'application/json':
        ModelsService().train_model(id, request.json)
        return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    else:
        return json.dumps({'result': '415 Unsuported Media Type'}, default=json_util.default)


@models.route('/validate/<id>', methods=['POST'])
def validate_model(id):
    if request.headers['Content-Type'] == 'application/json':
        ModelsService().validate_model(id, request.json)
        return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    else:
        return json.dumps({'result': '415 Unsuported Media Type'}, default=json_util.default)
