# Flame

Flame é um sistema de detecção de fraudes por imagens.
O processo consiste em classificar tipos de danos e reuso de peças danificadas por região.

# Testes Unitários

`python -m unittest`

# Executar o Servidor

## Windows 10

`pip install -r requirements.txt`

`$env:FLASK_APP="main:create_api"`
`$env:FLASK_ENV="development"`
`$env:FLASK_DEBUG=1`
`python -m flask run`
