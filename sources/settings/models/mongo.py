from sources.settings.configuration import CONFIGURATION


class MongoSettings(object):
    protocol = None
    host = None
    port = None
    user = None
    password = None
    database = None

    def __init__(self):
        self.mongo = CONFIGURATION['mongo']

        self.protocol = self.mongo['protocol']
        self.host = self.mongo['host']
        self.port = self.mongo['port']
        self.user = self.mongo['user']
        self.password = self.mongo['password']
        # self.database = self.mongo['database']
        self.database = 'flame'
