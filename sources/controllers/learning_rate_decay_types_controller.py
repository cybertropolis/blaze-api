import json

from flask import Blueprint
from bson import json_util
from bson.objectid import ObjectId

from sources.services.learning_rate_decay_types_service import LearningRateDecayTypesService

learning_rate_decay_types = Blueprint(
    'learning_rate_decay_types', __name__, url_prefix='/api/v1/learning_rate_decay_types')


@learning_rate_decay_types.route('/', methods=['GET'])
def get_models():
    results = LearningRateDecayTypesService().get_learning_rate_decay_types()
    json_results = []
    for result in results:
        json_results.append(result)
    return json.dumps(json_results, default=json_util.default)
