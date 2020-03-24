from flask import Flask
from flask_cors import CORS

from sources.controllers import (models_controller,
                                 neural_networks_controller,
                                 configurations_controller,
                                 learning_rate_decay_types_controller,
                                 optimizers_controller)


def create_api(test_config=None):

    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(models_controller.models)
    app.register_blueprint(neural_networks_controller.neural_networks)
    app.register_blueprint(configurations_controller.configurations)
    app.register_blueprint(
        learning_rate_decay_types_controller.learning_rate_decay_types)
    app.register_blueprint(optimizers_controller.optimizers)

    app.add_url_rule('/', endpoint="index")

    return app
