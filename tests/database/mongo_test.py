from unittest import TestCase, main
from mockupdb import MockupDB

from sources.databases.mongo import Mongo


class MongoTest(TestCase):

    def setUp(self):
        self.server = MockupDB(auto_ismaster=True, verbose=True)
        self.server.run()

        self.mongo = Mongo(self.server.uri).get_database()

    def test_connection(self):
        self.assertIsNotNone(self.mongo)


if __name__ == "__main__":
    main()
