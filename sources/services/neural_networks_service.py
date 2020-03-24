import logging

from bson.objectid import ObjectId

from sources.databases.mongo import Mongo
from sources.settings.models.mongo import MongoSettings


class NeuralNetworksService(object):

    def __init__(self):
        self.mongo_database = Mongo(MongoSettings()).get_database()

    def get_neural_networks(self):
        return self.mongo_database.neural_networks.find()

    def get_neural_network(self, id):
        return self.mongo_database.neural_networks.find_one({'_id': ObjectId(id)})

    def get_neural_networks_status(self):
        return {
            'total': self.mongo_database.neural_networks.find().count()
        }
