import logging

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from sources.settings.models.mongo import MongoSettings


class Mongo(object):
    def __init__(self, mongo_settings: MongoSettings):
        self.mongo_settings = mongo_settings

    def get_database(self):
        try:
            mongo = MongoClient('%s:%s' % (
                self.mongo_settings.host, self.mongo_settings.port))

            return mongo[self.mongo_settings.database]

        except (ConnectionFailure, ServerSelectionTimeoutError):
            logging.error(
                'O servidor do MongoDB está fora de alcance. Verifique a suas credencias e a disponibilidade do servidor.')
