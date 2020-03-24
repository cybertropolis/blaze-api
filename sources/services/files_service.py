import logging
import socket

from matplotlib.image import imread
from boto3 import resource
from tempfile import NamedTemporaryFile
from paramiko import (SSHClient, AutoAddPolicy,
                      AuthenticationException, BadHostKeyException, SSHException)

from sources.settings.models.storage import StorageSettings


class FileService(object):

    def __init__(self, storage_settings: StorageSettings):
        self.ssh = SSHClient()
        self.ssh.set_missing_host_key_policy(AutoAddPolicy())
        self.ssh.connect(
            hostname='%s://%s' % (storage_settings.protocol,
                                  storage_settings.host),
            username=storage_settings.user,
            port=storage_settings.port,
            password=storage_settings.password)

    def upload_file(self, local_file, remote_path, callback=None):
        try:
            sftp = self.ssh.open_sftp()
            with sftp.open(remote_path, 'wb') as remote_file:
                remote_file.write(local_file.read())
                remote_file.close()
            sftp.close()
        except (BadHostKeyException, AuthenticationException, SSHException, socket.error) as exception:
            logging.error(exception)

    def download_file(self, filename, remote_path, callback=None):
        try:
            sftp = self.ssh.open_sftp()
            with sftp.open(filename, 'rb') as remote_file:
                local_file = remote_file.read()
                remote_file.close()
            sftp.close()
            return local_file
        except (BadHostKeyException, AuthenticationException, SSHException, socket.error) as exception:
            logging.error(exception)

    def remove_file(self, filename):
        try:
            sftp = self.ssh.open_sftp()
            sftp.remove(filename)
            sftp.close()
        except (BadHostKeyException, AuthenticationException, SSHException, socket.error) as exception:
            logging.error(exception)

    def open_file(self):
        s3 = resource('s3', region_name='us-east-2')
        bucket = s3.Bucket('brain-spark')
        bucket_file = bucket.Object(
            'models/parts_identification/test/0000001.jpg')
        temporary = NamedTemporaryFile()

        with open(temporary, 'wb') as binary:
            bucket_file.download_file(binary)
            image = imread(temporary.name)

        return image
