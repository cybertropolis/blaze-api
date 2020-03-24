import tensorflow as tf

""" Usado para definir o fluxo de execução da aplicação de acordo com o ambiente.

Use o parâmetro --environment=develop para ambientes de desenvolvimento.
Use o parâmetro --environment=staging para ambientes de teste.
Use o parâmetro --environment=production para o ambiente de produção.
"""
tf.app.flags.DEFINE_string('environment', 'develop',
                           'Define qual é o ambiente usado pela rede neural.')

FLAGS = tf.app.flags.FLAGS
