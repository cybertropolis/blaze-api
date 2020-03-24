""" Generic training script that trains a model using a given dataset. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from bson.objectid import ObjectId

from sources.utilities.dataset_treatment import dataset_factory

from sources.services.deployment import model_deploy
from sources.services.neural_networks import neural_network_factory
from sources.services.preprocessing import preprocessing_factory

from sources.databases.mongo import Mongo
from sources.settings.models.mongo import MongoSettings

slim = tf.contrib.slim


class ImageClassifier(object):
    def __init__(self, model_id):
        mongo_database = Mongo(MongoSettings()).get_database()

        self.model = mongo_database.models.find_one(
            {'_id': ObjectId(model_id)})

        self.optimizer = mongo_database.optimizers.find_one(
            {'_id': ObjectId(self.model.optimization.optimizer_id)})

        self.learning_rate_decay_type = mongo_database.learning_rate_decay_types.find_one(
            {'_id': self.model.learning_rate.learning_rate_decay_type_id})

        self.neural_network = mongo_database.neural_networks.find_one(
            {'_id': ObjectId(self.model.neural_network_id)})

        if self.model.situations.testing:
            if not self.model.inputs.test:
                raise ValueError(
                    'Você deve fornecer um diretório de testes válido.')
            else:
                self.train_path = self.model.inputs.test_path

        if self.model.situations.training:
            if not self.model.inputs.train:
                raise ValueError(
                    'Você deve fornecer um diretório de treino válido.')
            else:
                self.train_path = self.model.inpust.train_path

        if self.model.situations.validating:
            if not self.model.inputs.validation:
                raise ValueError(
                    'Você deve fornecer um diretório de validação válido.')
            else:
                self.train_path = self.model.inpust.validation_path

    def _configure_learning_rate(self, samples_per_epoch, global_step):
        """ Configura a taxa de aprendizado.

        Argumentos:
            samples_per_epoch: O número de amostras em cada época de treinamento.
            global_step: o tensor global_step.

        Retornos:
            A `Tensor` representing the learning rate.

        Exceções:
            ValueError: Se o `self.learning_rate_decay_type.type` não for reconhecido.
        """

        # Nota: quando num_clones for > 1, isso fará com que cada clone repasse cada época
        # self.model.learning_rate.epochs_per_decay. Este é um comportamento diferente das
        # réplicas de sincronização e espera-se que produza resultados diferentes.
        decay_steps = int(samples_per_epoch * self.model.learning_rate.epochs_per_decay /
                          self.model.dataset.batch_size)

        if self.model.learning_rate.sync_replicas:
            decay_steps /= self.model.learning_rate.replicas_to_aggregate

        if self.learning_rate_decay_type.name == 'exponential':
            return tf.train.exponential_decay(self.model.learning_rate.exponential.learning_rate,
                                              global_step,
                                              decay_steps,
                                              self.model.learning_rate.exponential.decay_factor,
                                              staircase=self.model.learning_rate.exponential.staircase,
                                              name='exponential_decay_learning_rate')

        elif self.learning_rate_decay_type.name == 'fixed':
            return tf.constant(self.model.learning_rate.fixed.learning_rate, name='fixed_learning_rate')

        elif self.learning_rate_decay_type.name == 'polynomial':
            return tf.train.polynomial_decay(self.model.learning_rate.polynomial.learning_rate,
                                             global_step,
                                             decay_steps,
                                             self.model.learning_rate.polynomial.end_learning_rate,
                                             power=self.model.learning_rate.polynomial.power,
                                             cycle=self.model.learning_rate.polynomial.cycle,
                                             name='polynomial_decay_learning_rate')

        else:
            raise ValueError('O tipo de decaimento da taxa de aprendizagem "%s" não foi reconhecido.' %
                             self.learning_rate_decay_type.type)

    def _configure_optimizer(self, learning_rate):
        """ Configura o otimizador usado para treinamento.

        Argumentos:
        learning_rate: Uma taxa de aprendizado escalar ou `Tensor`.

        Retornos:
        Uma instância de um otimizador.

        Exceções:
        ValueError: Se `self.optimizer.name` não for reconhecido.
        """
        if self.optimizer.name == 'adadelta':
            return tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=self.model.optmization.optimizer.adadelta.rho,
                epsilon=self.model.optmization.optimizer.adadelta.optimizer_epsilon,
                name="adadelta_optimizer")

        elif self.optimizer.name == 'adagrad':
            return tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=self.model.optmization.optimizer.adagrad.initial_accumulator_value,
                name="adagrad_optimizer")

        elif self.optimizer.name == 'adam':
            return tf.train.AdamOptimizer(
                learning_rate,
                beta1=self.model.optmization.optimizer.adam.beta1,
                beta2=self.model.optmization.optimizer.adam.beta2,
                epsilon=self.model.optmization.optimizer.adam.optimizer_epsilon,
                name="adam_optimizer")

        elif self.optimizer.name == 'ftrl':
            return tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=self.model.optmization.optimizer.ftrl.learning_rate_power,
                initial_accumulator_value=self.model.optmization.optimizer.ftrl.initial_accumulator_value,
                l1_regularization_strength=self.model.optmization.optimizer.ftrl.l1,
                l2_regularization_strength=self.model.optmization.optimizer.ftrl.l2,
                name="ftrl_optimizer")

        elif self.optimizer.name == 'momentum':
            return tf.train.MomentumOptimizer(
                learning_rate,
                momentum=self.model.optmization.optimizer.momentum.momentum,
                name='momentum_optimizer')

        elif self.optimizer.name == 'rmsprop':
            return tf.train.RMSPropOptimizer(
                learning_rate,
                decay=self.model.optmization.optimizer.rmsprop.decay,
                momentum=self.model.optmization.optimizer.rmsprop.momentum,
                epsilon=self.model.optmization.optimizer.rmsprop.optimizer_epsilon,
                name="rmsprop_optimizer")
        elif self.optimizer.name == 'sgd':
            return tf.train.GradientDescentOptimizer(
                learning_rate,
                name="gradient_descent_optimizer")

        else:
            raise ValueError(
                'O otimizador "%s" não foi reconhecido.' % self.optimizer.name)

    def _get_init_fn(self):
        """ Retorna uma função executada pelo supervisor para iniciar o aquecimento do treinamento.

        Note que o init_fn é executado somente ao inicializar o modelo durante o primeiro passo global.

        Retornos:
        Uma função init executada pelo supervisor.
        """
        if self.model.fine_tuning.checkpoint_path is None:
            return None

        # Avisa o usuário se existe um ponto de verificação no train_path.
        # Então nós estaremos ignorando o checkpoint de qualquer maneira.
        if tf.train.latest_checkpoint(self.train_path):
            tf.logging.info(
                'Ignorando o caminho do checkpoint porque um ponto de verificação %s já existe'
                % self.train_path)
            return None

        exclusions = []
        if self.model.fine_tuning.checkpoint_excluded_paths:
            exclusions = [scope.strip()
                          for scope in self.model.fine_tuning.checkpoint_excluded_paths]

        variables_to_restore = []
        for variable in slim.get_model_variables():
            for exclusion in exclusions:
                if variable.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(variable)

        if tf.gfile.IsDirectory(self.model.fine_tuning.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(
                self.model.fine_tuning.checkpoint_path)
        else:
            checkpoint_path = self.model.fine_tuning.checkpoint_path

        tf.logging.info('Ajuste Fino do %s' % checkpoint_path)

        return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_variables=self.model.fine_tuning.ignore_missing_variables)

    def _get_variables_to_train(self):
        """ Retorna uma lista de variáveis para treinar.

        Retornos:
        Uma lista de variáveis para treinar pelo otimizador.
        """
        if self.model.fine_tuning.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip()
                      for scope in self.model.fine_tuning.trainable_scopes]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def train_model(self):

        tf.logging.set_verbosity(tf.logging.INFO)

        graph = tf.Graph()

        with tf.Session(graph=graph):
            # Configura o model_deploy
            deploy_config = model_deploy.DeploymentConfig(
                num_clones=self.model.device.num_clones,
                clone_on_cpu=self.model.device.clone_on_cpu,
                replica_id=self.model.device.task,
                num_replicas=self.model.device.worker_replicas,
                num_ps_tasks=self.model.device.num_ps_tasks)

            # Cria o global_step
            with tf.device(deploy_config.variables_device()):
                global_step = slim.create_global_step()

            # TODO: Modificar
            # Seleciona o dataset
            dataset = dataset_factory.get_dataset(
                self.model.dataset.name, self.model.dataset.split_name, self.model.dataset.directory)

            # TODO: Modificar
            # Seleciona a rede neural
            network_fn = neural_network_factory.get_network_fn(
                self.neural_network.name,
                num_classes=(dataset.num_classes -
                             self.model.dataset.labels_offset),
                weight_decay=self.model.optimization.weight_decay,
                is_training=True)

            # TODO: Modificar
            # Seleciona a função de preprocessamento
            preprocessing_name = self.neural_network.preprocessing_name or self.neural_network.name
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=True)

            # Criar um provedor de conjunto de dados que carrega os dados de um conjunto de dados
            with tf.device(deploy_config.inputs_device()):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=self.model.dataset.num_readers,
                    common_queue_capacity=20 * self.model.dataset.batch_size,
                    common_queue_min=10 * self.model.dataset.batch_size)
                [image, label] = provider.get(['image', 'label'])
                label -= self.model.dataset.labels_offset

                train_image_size = (
                    self.model.dataset.train_image_size if self.model.dataset.train_image_size else network_fn.default_image_size)

                image = image_preprocessing_fn(
                    image, train_image_size, train_image_size)

                images, labels = tf.train.batch(
                    [image, label],
                    batch_size=self.model.dataset.batch_size,
                    num_threads=self.model.dataset.preprocessing_threads_number,
                    capacity=5 * self.model.dataset.batch_size)

                labels = slim.one_hot_encoding(
                    labels, dataset.num_classes - self.model.dataset.labels_offset)

                batch_queue = slim.prefetch_queue.prefetch_queue(
                    [images, labels], capacity=2 * deploy_config.num_clones)

            # Define o Modelo
            def clone_fn(batch_queue):
                """ Permite o paralelismo de dados criando múltiplos clones de network_fn. """
                images, labels = batch_queue.dequeue()
                logits, end_points = network_fn(images)

                # Especifica a função de perda (loss function)
                if 'AuxLogits' in end_points:
                    slim.losses.softmax_cross_entropy(
                        end_points['AuxLogits'],
                        labels,
                        label_smoothing=self.model.learning_rate.label_smoothing,
                        weights=0.4,
                        scope='aux_loss')

                slim.losses.softmax_cross_entropy(
                    logits,
                    labels,
                    label_smoothing=self.model.learning_rate.label_smoothing,
                    weights=1.0)
                    
                return end_points

            # Reune os sumários (summaries) iniciais.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            clones = model_deploy.create_clones(
                deploy_config, clone_fn, [batch_queue])

            first_clone_scope = deploy_config.clone_scope(0)

            # Gather update_ops from the first clone. These contain, for example,
            # the updates for the batch_norm variables created by network_fn.
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, first_clone_scope)

            # Adiciona os sumários (summaries) nos end_points.
            end_points = clones[0].outputs
            for end_point in end_points:
                x = end_points[end_point]
                summaries.add(tf.summary.histogram(
                    'activations/' + end_point, x))
                summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                                tf.nn.zero_fraction(x)))

            # Adiciona sumários (summaries) para perdas (losses).
            for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
                summaries.add(tf.summary.scalar(
                    'losses/%s' % loss.op.name, loss))

            # Adiciona resumos (summaries) para variáveis (variables).
            for variable in slim.get_model_variables():
                summaries.add(tf.summary.histogram(variable.op.name, variable))

            # Configure as médias móveis (moving averages).
            if self.model.learning_rate.moving_average_decay:
                moving_average_variables = slim.get_model_variables()
                variable_averages = tf.train.ExponentialMovingAverage(
                    self.model.learning_rate.moving_average_decay, global_step)
            else:
                moving_average_variables, variable_averages = None, None

            # Configura o procedimento de otimização.
            with tf.device(deploy_config.optimizer_device()):
                learning_rate = self._configure_learning_rate(
                    dataset.num_samples, global_step)
                optimizer = self._configure_optimizer(learning_rate)
                summaries.add(tf.summary.scalar(
                    'learning_rate', learning_rate))

            if self.model.learning_rate.sync_replicas:
                # Se sync_replicas estiver ativado, a média será feita no gerenciador de filas principal.
                optimizer = tf.train.SyncReplicasOptimizer(
                    opt=optimizer,
                    replicas_to_aggregate=self.model.learning_rate.replicas_to_aggregate,
                    total_num_replicas=self.model.device.worker_replicas,
                    variable_averages=variable_averages,
                    variables_to_average=moving_average_variables)
            elif self.model.learning_rate.moving_average_decay:
                # Atualiza as operações executadas localmente pelo treinador.
                update_ops.append(variable_averages.apply(
                    moving_average_variables))

            # Variáveis para treinar.
            variables_to_train = self._get_variables_to_train()

            # Retorna um train_tensor e um summary_op
            total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                optimizer,
                var_list=variables_to_train)

            # Adiciona a perda total (total_loss) ao sumário (summary).
            summaries.add(tf.summary.scalar('total_loss', total_loss))

            # Cria atualizações de gradiente.
            gradient_updates = optimizer.apply_gradients(clones_gradients,
                                                         global_step=global_step)
            update_ops.append(gradient_updates)

            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')

            # Adicione os sumários do primeiro clone.
            # Estes contêm os sumários criados pelo model_fn() e otimize_clones() ou _gather_clone_loss().
            summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                               first_clone_scope))

            # Mescla todos os sumários (summaries) juntos.
            summary_op = tf.summary.merge(list(summaries), name='summary_op')

            # Inicia o treinamento.
            slim.learning.train(
                train_tensor,
                logdir=self.train_path,
                master=self.model.device.master,
                is_chief=(self.model.device.task == 0),
                init_fn=self._get_init_fn(),
                summary_op=summary_op,
                number_of_steps=self.model.dataset.max_number_of_steps if self.model.dataset.max_number_of_steps else None,
                log_every_n_steps=self.model.log_every_n_steps,
                save_summaries_secs=self.model.save_summaries_secs,
                save_interval_secs=self.model.save_interval_secs,
                sync_optimizer=optimizer if self.model.learning_rate.sync_replicas else None)
