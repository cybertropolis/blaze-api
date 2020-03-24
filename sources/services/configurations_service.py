import logging

from bson import BSON
from bson.objectid import ObjectId

from sources.databases.mongo import Mongo
from sources.settings.models.mongo import MongoSettings


class ConfigurationsService(object):

    def __init__(self):
        self.mongo_database = Mongo(MongoSettings()).get_database()

    def get_configurations(self):
        return self.mongo_database.configurations.find_one({'_id': 'configurations'})

    def save_configurations(self, configurations):
        return self.mongo_database.configurations.save(configurations)
