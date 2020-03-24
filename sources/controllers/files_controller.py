from flask import Flask

blue_print = Flask('files', __name__)


@blue_print.route('/', methods=['POST'])
def upload():
    return


@blue_print.route('/', methods=['GET'])
def download():
    return 'Arquivos enviados com sucesso!'
