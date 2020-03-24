import logging

from bson.objectid import ObjectId

from sources.databases.mongo import Mongo
from sources.settings.models.mongo import MongoSettings


class ModelsService(object):

    def __init__(self):
        self.mongo_database = Mongo(MongoSettings()).get_database()

    def get_models(self):
        return self.mongo_database.models.find()

    def get_model(self, id):
        return self.mongo_database.models.find_one({'_id': ObjectId(id)})

    def get_models_status(self):
        return {
            'total': self.mongo_database.models.find().count()
        }

    def create_model(self, model):
        return self.mongo_database.models.insert_one(model)

    def update_model(self, id, model):
        return self.mongo_database.models.update_one({'_id': ObjectId(id)}, {
            '$set': model
        })

    def test_model(self, id, model):
        return self.mongo_database.models.update_one({'_id': ObjectId(id)}, {
            '$set': {
                'situations.testing': model['situations']['testing']
            }
        })

    def train_model(self, id, model):
        return self.mongo_database.models.update_one({'_id': ObjectId(id)}, {
            '$set': {
                'situations.training': model['situations']['training']
            }
        })

    def validate_model(self, id, model):
        return self.mongo_database.models.update_one({'_id': ObjectId(id)}, {
            '$set': {
                'situations.validating': model['situations']['validating']
            }
        })
