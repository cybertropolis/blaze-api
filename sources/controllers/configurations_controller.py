import json

from flask import Blueprint, json, request
from bson import json_util
from bson.objectid import ObjectId

from sources.services.configurations_service import ConfigurationsService

configurations = Blueprint(
    'configurations', __name__, url_prefix='/api/v1/configurations')


@configurations.route('/', methods=['GET'])
def get_configurations():
    result = ConfigurationsService().get_configurations()
    return json.dumps(result, default=json_util.default)


@configurations.route('/save', methods=['POST', 'PUT'])
def save_configurations():
    if request.headers['Content-Type'] == 'application/json':
        response = ConfigurationsService().save_configurations(request.json)
        return json.dumps(response, default=json_util.default)
    else:
        return json.dumps({'result': '415 Unsuported Media Type'}, default=json_util.default)
