from sources.settings.configuration import CONFIGURATION


class StorageSettings(object):
    protocol = None
    host = None
    port = None
    user = None
    password = None
    local_path = None
    remote_path = None

    def __init__(self):
        storage = CONFIGURATION['storage']

        self.protocol = storage['protocol']
        self.host = storage['host']
        self.port = storage['port']
        self.user = storage['user']
        self.password = storage['password']
        self.local_path = storage['local_path']
        self.remote_path = storage['remote_path']
