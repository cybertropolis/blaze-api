import json

from flask import Blueprint
from bson import json_util
from bson.objectid import ObjectId

from sources.services.optimizers_service import OptimizersService

optimizers = Blueprint(
    'optimizers', __name__, url_prefix='/api/v1/optimizers')


@optimizers.route('/', methods=['GET'])
def get_optimizers():
    results = OptimizersService().get_optimizers()
    json_results = []
    for result in results:
        json_results.append(result)
    return json.dumps(json_results, default=json_util.default)
