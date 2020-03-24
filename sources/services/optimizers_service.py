import logging

from bson.objectid import ObjectId

from sources.databases.mongo import Mongo
from sources.settings.models.mongo import MongoSettings


class OptimizersService(object):

    def __init__(self):
        self.mongo_database = Mongo(MongoSettings()).get_database()

    def get_optimizers(self):
        return self.mongo_database.optimizers.find()
