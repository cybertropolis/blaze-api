from unittest import TestCase, main
from mockupdb import MockupDB

from sources.services.file_service import FileService
from sources.settings.models.storage import StorageSettings


class FileServiceTest(TestCase):

    def setUp(self):
        self._file_service = FileService(StorageSettings())

    def test_connection(self):
        self.assertIsNotNone(self._file_service)


if __name__ == '__main__':
    main()
