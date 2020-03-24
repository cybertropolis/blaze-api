import os

from sources.settings.environment import FLAGS

CONFIGURATION = {
    'storage': {
        'protocol': 'ftp',
        'host': os.environ.get('STORAGE_HOST_%s' % FLAGS.environment.upper()),
        'port': os.environ.get('STORAGE_PORT_%s' % FLAGS.environment.upper()),
        'user': os.environ.get('STORAGE_USER_%s' % FLAGS.environment.upper()),
        'password': os.environ.get('STORAGE_PASSWORD_%s' % FLAGS.environment.upper()),
        'local_path': os.environ.get('STORAGE_LOCAL_PATH_%s' % FLAGS.environment.upper()),
        'remote_path': os.environ.get('STORAGE_REMOTE_PATH_%s' % FLAGS.environment.upper())
    },
    'mongo': {
        'protocol': 'mongodb',
        'host': os.environ.get('MONGO_HOST_%s' % FLAGS.environment.upper()),
        'port': os.environ.get('MONGO_PORT_%s' % FLAGS.environment.upper()),
        'user': os.environ.get('MONGO_USER_%s' % FLAGS.environment.upper()),
        'password': os.environ.get('MONGO_PASSWORD_%s' % FLAGS.environment.upper()),
        'database': os.environ.get('MONGO_DATABASE_%s' % FLAGS.environment.upper())
    }
}
