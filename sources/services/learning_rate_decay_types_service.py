import logging

from bson.objectid import ObjectId

from sources.databases.mongo import Mongo
from sources.settings.models.mongo import MongoSettings


class LearningRateDecayTypesService(object):

    def __init__(self):
        self.mongo_database = Mongo(MongoSettings()).get_database()

    def get_learning_rate_decay_types(self):
        return self.mongo_database.learning_rate_decay_types.find()
