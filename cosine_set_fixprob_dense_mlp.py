# This is pre-alpha free software and was tested in Ubuntu 16.04 with Python 3.5.2, Numpy 1.15.4, Sklearn 0.19.1, TensorFlow 1.9, Keras 2.0.8, Keras Contrib 0.0.2.

from __future__ import division
from __future__ import print_function

import json

import numpy as np

import sys
import os

sys.path.append(os.getcwd())

# Cython
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

# Install the plaidml backend
# import plaidml.keras
# plaidml.keras.install_backend()
import datetime

from keras.backend import sigmoid, categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers, losses
from keras import backend as K
from keras_contrib.layers.advanced_activations import SReLU
from keras.utils import np_utils
from sklearn.metrics.pairwise import cosine_distances

from scipy.io import loadmat
import os
from scipy import spatial
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib

import networkx as nx
from networkx.algorithms import bipartite

# extra imports to set GPU options
import tensorflow as tf

config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.3

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

start_time = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
result_folder = "results/run_" + start_time + "/"
print("result_folder", result_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


class Constraint(object):
    def __call__(self, w):
        return w

    def get_config(self):
        return {}


class MaskWeights(Constraint):
    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}


class Generator:
    def flow(self, x_train, y_train, batch_size):
        data_size = len(x_train)
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        while True:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = x_train[shuffle_indices]
            shuffled_labels = y_train[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                batch_inputs = shuffled_data[start_index: end_index]
                batch_labels = shuffled_labels[start_index: end_index]
                yield batch_inputs, batch_labels


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createWeightsMask(epsilon, noRows, noCols):
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print("Create Sparse Matrix: No parameters, NoRows, NoCols ", noParameters, noRows, noCols)
    return [noParameters, mask_weights]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))


def degrees(weights):
    return [sum(1 if weight != 0 else 0 for weight in weight_row) for weight_row in weights]


def weighted_graph_from_weight_masks(weight_masks):
    nodes = []
    nodes.append(["0-{}".format(i) for i in range(weight_masks[0].shape[0])])
    graph = nx.Graph()
    graph.add_nodes_from(nodes[0])
    for layer_number, weight_mask in enumerate(weight_masks):
        left, right = weight_mask.shape
        nodes.append(["{}-{}".format(layer_number + 1, i) for i in range(right)])
        graph.add_nodes_from(nodes[layer_number + 1])
        graph.add_weighted_edges_from(
            [(nodes[layer_number][i], nodes[layer_number + 1][j], weight_mask[i][j]) for i in range(left) for j in
             range(right)])
    return graph


def unweighted_graph_from_weight_masks(weight_masks):
    nodes = []
    nodes.append(["0-{}".format(i) for i in range(weight_masks[0].shape[0])])
    graph = nx.Graph()
    graph.add_nodes_from(nodes[0])
    for layer_number, weight_mask in enumerate(weight_masks):
        left, right = weight_mask.shape
        nodes.append(["{}-{}".format(layer_number + 1, i) for i in range(right)])
        graph.add_nodes_from(nodes[layer_number + 1])
        graph.add_edges_from(
            [(nodes[layer_number][i], nodes[layer_number + 1][j]) for i in range(left) for j in range(right) if
             weight_mask[i][j] != 0])
    return graph


class mlp:
    def __init__(self, data, num_classes, input_shape, settings):
        """
        :param data: [x_train, y_train, x_test, y_test]
        :param typemlp: 
            dense: regular ANN
            fixprob: ANN with fixed topology
            evolutionary: SET-MLP, so dynamic topology
        :param num_classes: the number of output classes
        :param input_shape: e.g. (1, 28, 28)
        :param learning_rate: 
        :param hidden_layer_sizes: list of number of neurons for each hidden layer
        :param is_image: indicates if the data consists of image or not 
        :param batch_size: 
        :param maxepoches: 
        :param zeta: num_weights to be removed / total number of weights
        """
        self.data = data
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.settings = settings
        self.learning_rate = settings['learning_rate']
        self.typemlp = settings['typemlp']
        self.hidden_layer_sizes = settings['hidden_layer_sizes']
        self.use_centralities = settings['use_centralities']
        self.is_image = settings['is_image']
        self.epsilon = settings['epsilon']
        self.sim_sample_size = settings.get('sim_sample_size') or 0.0
        self.center_sim = settings.get('center_sim') or False
        self.remove_using_sim = settings.get('remove_using_sim') or False
        self.remove_using_weighted_sim = settings.get('remove_using_weighted_sim') or False
        self.add_using_sim = settings.get('add_using_sim') or False
        self.randomly_add_using_sim = settings.get('randomly_add_using_sim') or False
        self.batch_size = settings.get('batch_size') or 100
        self.maxepoches = settings.get('maxepoches') or 1000
        self.zeta = settings.get('zeta') or 0.3
        self.pref_attach_alpha = settings.get('pref_attach_alpha') or 0
        if self.typemlp == "evolutionary" or self.typemlp == "fixprob":
            [self.noPar1, self.wm1] = createWeightsMask(self.epsilon, np.prod(np.array(self.input_shape)),
                                                        self.hidden_layer_sizes[0])
            [self.noPar2, self.wm2] = createWeightsMask(self.epsilon, self.hidden_layer_sizes[0],
                                                        self.hidden_layer_sizes[1])
            [self.noPar3, self.wm3] = createWeightsMask(self.epsilon, self.hidden_layer_sizes[1],
                                                        self.hidden_layer_sizes[2])

        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

        self.wSRelu1 = None
        self.wSRelu2 = None
        self.wSRelu3 = None

        self.similarities = (None, None, None)

        self.create_model()
        self.train()
        if self.typemlp == "evolutionary":
            self.benchmark_feature_selection("final")

    def rewireMask(self, weights, layer_number, edge_centrality, similarities, noPar, epoch):
        if self.use_centralities:
            edge_centrality_matrix = np.zeros(weights.shape)
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    edge_centrality_matrix[i][j] = edge_centrality.get(
                        ("{}-{}".format(layer_number, i), "{}-{}".format(layer_number + 1, j))) or \
                                                   edge_centrality.get(("{}-{}".format(layer_number + 1, j),
                                                                        "{}-{}".format(layer_number, i)), 0)
            centralitied_weights = np.multiply(weights, edge_centrality_matrix)
        else:
            centralitied_weights = weights

        sim_flattened_probabilities = []
        sim_indices = []
        if similarities is not None:
            sim_indices = [np.unravel_index(p, similarities.shape) for p in np.argsort(similarities, axis=None)[::-1]]
            print("first", sim_indices[0], similarities[sim_indices[0]], "last", sim_indices[-1],
                  similarities[sim_indices[-1]])

        rewiredWeights = centralitied_weights.copy()
        if self.remove_using_sim:
            sim_threshold = similarities[sim_indices[int(self.zeta * noPar)]]
            for i, node_weights in enumerate(rewiredWeights):
                for j, _ in enumerate(node_weights):
                    if similarities[i][j] < sim_threshold:
                        rewiredWeights[i][j] = 0
                    else:
                        rewiredWeights[i][j] = 1
        else:
            rewiredWeights = centralitied_weights.copy()
            if self.remove_using_weighted_sim:
                rewiredWeights *= similarities
            values = np.sort(rewiredWeights.ravel())
            firstZeroPos = find_first_pos(values, 0)
            lastZeroPos = find_last_pos(values, 0)
            largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
            smallestPositive = values[
                int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

            rewiredWeights[rewiredWeights > smallestPositive] = 1
            rewiredWeights[rewiredWeights < largestNegative] = 1
            rewiredWeights[rewiredWeights != 1] = 0
        weightMaskCore = rewiredWeights.copy()

        pref_degrees = []
        pref_transposed_degrees = []
        if self.pref_attach_alpha != 0:
            pref_degrees = [(degree ** self.pref_attach_alpha + 1) for degree in degrees(centralitied_weights)]
            sum_degrees = sum(pref_degrees) + len(pref_degrees)
            pref_degrees = [(degree + 1) / sum_degrees for degree in pref_degrees]

            pref_transposed_degrees = [(transposed_degree ** self.pref_attach_alpha + 1) for transposed_degree in
                                       degrees(centralitied_weights.T)]
            sum_transposed_degrees = sum(pref_transposed_degrees) + len(pref_transposed_degrees)
            pref_transposed_degrees = [(transposed_degree + 1) / sum_transposed_degrees for transposed_degree in
                                       pref_transposed_degrees]

        nre = 0
        noRewires = noPar - np.sum(rewiredWeights)
        if self.add_using_sim:
            for index in sim_indices:
                if rewiredWeights[index] == 0:
                    rewiredWeights[index] = 1
                    nre += 1
                if nre >= noRewires:
                    break
        if self.randomly_add_using_sim:
            sample_index = 0
            total_sim_sum = np.sum(similarities)
            sim_flattened_probabilities = np.array(
                [similarities[i] / total_sim_sum for i in range(len(similarities))]).flatten()
            sim_flattened_indices_sample = np.random.choice(weights.shape[0] * weights.shape[1],
                                                            int(noRewires / self.zeta), p=sim_flattened_probabilities,
                                                            replace=False)

        while nre < noRewires:
            if self.randomly_add_using_sim:
                sample_index += 1
                if sample_index < len(sim_flattened_indices_sample):
                    flattened_index = sim_flattened_indices_sample[sample_index]
                else:
                    flattened_index = np.random.choice(rewiredWeights.shape[0] * rewiredWeights.shape[1],
                                                       p=sim_flattened_probabilities)
                i = flattened_index // rewiredWeights.shape[1]
                j = flattened_index % rewiredWeights.shape[1]
            elif self.pref_attach_alpha != 0:
                i = np.random.choice(rewiredWeights.shape[0], p=pref_degrees)
                j = np.random.choice(rewiredWeights.shape[1], p=pref_transposed_degrees)
            else:
                i = np.random.randint(0, rewiredWeights.shape[0])
                j = np.random.randint(0, rewiredWeights.shape[1])
            if rewiredWeights[i, j] == 0:
                rewiredWeights[i, j] = 1
                nre += 1
        return [rewiredWeights, weightMaskCore]

    def create_model(self):
        K.set_learning_phase(1)
        self.model = Sequential()
        if len(self.input_shape) > 2:
            self.model.add(Flatten(input_shape=self.input_shape))

        if (self.typemlp == "dense"):
            if len(self.input_shape) <= 2:
                self.model.add(
                    Dense(self.hidden_layer_sizes[0], name="dense_1", weights=self.w1, input_shape=self.input_shape))
            else:
                self.model.add(Dense(self.hidden_layer_sizes[0], name="dense_1", weights=self.w1))
            self.model.add(SReLU(name="srelu1", weights=self.wSRelu1))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.hidden_layer_sizes[1], name="dense_2", weights=self.w2))
            self.model.add(SReLU(name="srelu2", weights=self.wSRelu2))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.hidden_layer_sizes[2], name="dense_3", weights=self.w3))
            self.model.add(SReLU(name="srelu3", weights=self.wSRelu3))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.num_classes, name="dense_4", weights=self.w4))
            self.model.add(Activation('softmax'))
        else:
            if len(self.input_shape) <= 2:
                self.model.add(
                    Dense(self.hidden_layer_sizes[0], name="sparse_1", kernel_constraint=MaskWeights(self.wm1),
                          weights=self.w1, input_shape=self.input_shape))
            else:
                self.model.add(
                    Dense(self.hidden_layer_sizes[0], name="sparse_1", kernel_constraint=MaskWeights(self.wm1),
                          weights=self.w1))
            self.model.add(SReLU(name="srelu1", weights=self.wSRelu1))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.hidden_layer_sizes[1], name="sparse_2", kernel_constraint=MaskWeights(self.wm2),
                                 weights=self.w2))
            self.model.add(SReLU(name="srelu2", weights=self.wSRelu2))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.hidden_layer_sizes[2], name="sparse_3", kernel_constraint=MaskWeights(self.wm3),
                                 weights=self.w3))
            self.model.add(SReLU(name="srelu3", weights=self.wSRelu3))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.num_classes, name="dense_4", weights=self.w4))
            self.model.add(Activation('softmax'))

    def save_weights_rewire(self, epoch):
        if (self.typemlp == "dense"):
            self.w1 = self.model.get_layer("dense_1").get_weights()
            self.w2 = self.model.get_layer("dense_2").get_weights()
            self.w3 = self.model.get_layer("dense_3").get_weights()
            self.w4 = self.model.get_layer("dense_4").get_weights()
        else:
            self.w1 = self.model.get_layer("sparse_1").get_weights()
            self.w2 = self.model.get_layer("sparse_2").get_weights()
            self.w3 = self.model.get_layer("sparse_3").get_weights()
            self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wSRelu1 = self.model.get_layer("srelu1").get_weights()
        self.wSRelu2 = self.model.get_layer("srelu2").get_weights()
        self.wSRelu3 = self.model.get_layer("srelu3").get_weights()

        if self.typemlp == "evolutionary":
            centrality = None
            if self.use_centralities:
                weight_masks = np.array([self.w1[0], self.w2[0], self.w3[0], self.w4[0]])
                print("Started creating graph at", datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
                graph = weighted_graph_from_weight_masks(weight_masks)
                print("Graph created, start calculating centrality",
                      datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
                centrality = nx.edge_current_flow_betweenness_centrality(graph)
                print("Centrality calculated", datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
                plt.figure()
                plt.hist(centrality.values(), bins=100)
                plt.title('Centrality distribution')
                plt.xlabel("value")
                plt.ylabel("Frequency")
                plt.savefig(result_folder + "centralities.png")
                plt.close()

            [self.wm1, self.wm1Core] = self.rewireMask(self.w1[0], 0, centrality, self.similarities[0], self.noPar1,
                                                       epoch)
            [self.wm2, self.wm2Core] = self.rewireMask(self.w2[0], 1, centrality, self.similarities[1], self.noPar2,
                                                       epoch)
            [self.wm3, self.wm3Core] = self.rewireMask(self.w3[0], 2, centrality, self.similarities[2], self.noPar3,
                                                       epoch)

            print(sum(sum(connections) for connections in self.wm1))

            self.w1[0] = self.w1[0] * self.wm1Core
            self.w2[0] = self.w2[0] * self.wm2Core
            self.w3[0] = self.w3[0] * self.wm3Core

    def read_data(self, x_train, y_train, x_test, y_test):
        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        xTrainMean = np.mean(x_train, axis=0)
        xTtrainStd = np.std(x_train, axis=0)
        print("x_train std shape", xTtrainStd.shape)
        # (x_train - xTrainMean) / xTrainMean, but prevent dividing by 0
        x_train = np.divide(x_train - xTrainMean, xTtrainStd, out=np.zeros_like(x_train - xTrainMean),
                            where=xTtrainStd != 0)
        # (x_test - xTrainMean) / xTrainMean, but prevent dividing by 0
        x_test = np.divide(x_test - xTrainMean, xTtrainStd, out=np.zeros_like(x_test - xTrainMean),
                           where=xTtrainStd != 0)

        return [x_train, x_test, y_train, y_test]

    def train(self):
        [self.x_train, self.x_test, self.y_train, self.y_test] = self.read_data(*self.data)

        if self.is_image:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # Alternatively, no augmentation:
            # datagen = ImageDataGenerator()
            datagen.fit(self.x_train)
        else:
            datagen = Generator()

        self.model.summary()

        # training process in a for loop
        all_accs = []
        for epoch in range(0, self.maxepoches):

            sgd = optimizers.SGD(lr=self.learning_rate, momentum=0.9)
            self.model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

            historytemp = self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                                                   steps_per_epoch=self.x_train.shape[0] // self.batch_size,
                                                   epochs=epoch,
                                                   validation_data=(self.x_test, self.y_test),
                                                   initial_epoch=epoch - 1,
                                                   verbose=2)

            all_accs.append(historytemp.history['val_acc'][0])

            if epoch == self.maxepoches - 1:
                break

            # if epoch in {1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900}:
            #     self.benchmark_feature_selection(epoch)

            if self.typemlp == "evolutionary":
                self.compute_similarities()
            # ugly hack to avoid tensorflow memory increase for multiple fit_generator calls. Theano shall work more nicely this but it is outdated in general
            self.save_weights_rewire(epoch)
            K.clear_session()
            self.create_model()

        self.accuracies_case = np.asarray(all_accs)

    def compute_similarities(self):
        if self.sim_sample_size <= 0:
            return
        print("start calculating activations")
        activation_start_time = datetime.datetime.now()

        activations = {}
        get_flattened_input = K.function([self.model.layers[0].input],
                                         [self.model.layers[0].output])
        get_1st_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.get_layer("sparse_1").output])
        get_2nd_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.get_layer("sparse_2").output])
        get_3rd_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.get_layer("sparse_3").output])
        get_4th_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.get_layer("dense_4").output])
        activations["sparse_1"] = get_1st_layer_output([self.x_train])[0]
        activations["sparse_2"] = get_2nd_layer_output([self.x_train])[0]
        activations["sparse_3"] = get_3rd_layer_output([self.x_train])[0]
        activations["dense_4"] = get_4th_layer_output([self.x_train])[0]

        print("calculating activations took", datetime.datetime.now() - activation_start_time)

        flattened_input = self.x_train
        if len(self.input_shape) > 2:
            flattened_input = get_flattened_input([self.x_train])[0]

        sim_start_time = datetime.datetime.now()
        # layer_1_similarities = similarities.layer_similarities(flattened_input.T.astype('float64'),
        #                                                        activations["sparse_1"].T.astype('float64'),
        #                                                        self.sim_sample_size, self.center_sim)
        # layer_2_similarities = similarities.layer_similarities(activations["sparse_1"].T.astype('float64'),
        #                                                        activations["sparse_2"].T.astype('float64'),
        #                                                        self.sim_sample_size, self.center_sim)
        # layer_3_similarities = similarities.layer_similarities(activations["sparse_2"].T.astype('float64'),
        #                                                        activations["sparse_3"].T.astype('float64'),
        #                                                        self.sim_sample_size, self.center_sim)
        layer_1_similarities = np.absolute(np.array(1 - cosine_distances(flattened_input.T, activations["sparse_1"].T)))
        layer_2_similarities = np.absolute(
            np.array(1 - cosine_distances(activations["sparse_1"].T, activations["sparse_2"].T)))
        layer_3_similarities = np.absolute(
            np.array(1 - cosine_distances(activations["sparse_2"].T, activations["sparse_3"].T)))

        self.similarities = (layer_1_similarities, layer_2_similarities, layer_3_similarities)
        print("calculating similarities for all layers took", datetime.datetime.now() - sim_start_time)

    def benchmark_feature_selection(self, epoch):
        num_weights_per_input = np.array([sum(connections) for connections in self.wm1Core])
        np.savetxt(result_folder + "num_weights_per_input.txt", np.asarray(num_weights_per_input),
                   header=json.dumps(self.settings, indent=4))
        least_to_most_connected = []
        get_flattened_input = K.function([self.model.layers[0].input], [self.model.layers[0].output])
        flattened_x_test = self.x_test
        # if len(self.input_shape) > 2:
        #     flattened_x_test = get_flattened_input([self.x_test])[0]
        x_test_copy = flattened_x_test.copy()
        for i in np.argsort(num_weights_per_input):
            # print(get_flattened_input([self.x_test])[0][1][i])
            # print(x_test_copy[1][int(i / self.input_shape[0])][i % self.input_shape[0]])
            if len(self.input_shape) > 2:
                for sample in x_test_copy:
                    sample[int(i / self.input_shape[0])][i % self.input_shape[0]] = 0
            else:
                x_test_copy.T[i] = 0
            least_to_most_connected.append(self.model.evaluate(x_test_copy, self.y_test, verbose=0)[1])
        np.savetxt(result_folder + "epoch" + str(epoch) + "_least_to_most_accuracies.txt",
                   np.asarray(least_to_most_connected),
                   header=json.dumps(self.settings, indent=4))
        # Gradually remove input neurons, starting with most connected ones
        most_to_least_connected = []
        x_test_copy = flattened_x_test.copy()
        for i in np.argsort(num_weights_per_input)[::-1]:
            if len(self.input_shape) > 2:
                for sample in x_test_copy:
                    sample[int(i / self.input_shape[0])][i % self.input_shape[0]] = 0
            else:
                x_test_copy.T[i] = 0
            most_to_least_connected.append(self.model.evaluate(x_test_copy, self.y_test, verbose=0)[1])
        np.savetxt(result_folder + "epoch" + str(epoch) + "_most_to_least_accuracies.txt",
                   np.asarray(most_to_least_connected),
                   header=json.dumps(self.settings, indent=4))


def test_fashion_mnist(default_settings):
    # https://github.com/zalandoresearch/fashion-mnist
    # Existing training/test split
    num_classes = 10
    input_shape = (28, 28, 1)
    default_settings['dataset'] = "Fashion-MNIST"
    default_settings['is_image'] = True
    default_settings['hidden_layer_sizes'] = [200, 200, 200]
    settings = default_settings
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    data = [x_train, y_train, x_test, y_test]
    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


def test_isolet(default_settings):
    # Speech recognition dataset
    # Best accuracy the creators obtained with a neural net: 95%
    # https://www.researchgate.net/profile/Ronald_Cole2/publication/224755052_Speaker-independent_recognition_of_spoken_English_letters/links/569ebe4908ae4af525449d81/Speaker-independent-recognition-of-spoken-English-letters.pdf
    # Existing training/test split
    num_classes = 26
    input_shape = (617,)
    default_settings['dataset'] = "ISOLET"
    default_settings['is_image'] = False
    default_settings['hidden_layer_sizes'] = [1000, 1000, 1000]
    settings = default_settings
    with open("datasets/ISOLET/isolet1+2+3+4.data") as train_file:
        lines = train_file.readlines()
        x_train = np.array([line.strip().split(", ")[:-1] for line in lines])
        y_train = np.array([int(line.strip().split(", ")[-1].replace(".", "")) - 1 for line in lines])
    with open("datasets/ISOLET/isolet5.data") as train_file:
        lines = train_file.readlines()
        x_test = np.array([line.strip().split(", ")[:-1] for line in lines])
        y_test = np.array([int(line.strip().split(", ")[-1].replace(".", "")) - 1 for line in lines])
    data = [x_train, y_train, x_test, y_test]
    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


def test_har(default_settings):
    # https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    # Existing training/test split
    input_shape = (561,)
    num_classes = 6
    default_settings['dataset'] = "Human Activity Recognition"
    default_settings['is_image'] = False
    default_settings['hidden_layer_sizes'] = [1000, 1000, 1000]
    settings = default_settings
    with open("datasets/har/train/X_train.txt") as x_train_file:
        lines = x_train_file.readlines()
        x_train = np.array([line.strip().split() for line in lines])
    with open("datasets/har/train/y_train.txt") as y_train_file:
        lines = y_train_file.readlines()
        y_train = np.array([int(line) - 1 for line in lines])
    with open("datasets/har/test/X_test.txt") as x_test_file:
        lines = x_test_file.readlines()
        x_test = np.array([line.strip().split() for line in lines])
    with open("datasets/har/test/y_test.txt") as y_test_file:
        lines = y_test_file.readlines()
        y_test = np.array([int(line) - 1 for line in lines])
    data = [x_train, y_train, x_test, y_test]
    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


def test_coil100(default_settings):
    # http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php
    # Custom training/test split
    num_classes = 100
    input_shape = (32, 32, 1)
    default_settings['dataset'] = "COIL100"
    default_settings['is_image'] = True
    default_settings['hidden_layer_sizes'] = [1000, 1000, 1000]
    settings = default_settings

    with open("datasets/COIL100/train.csv") as train_file:
        lines = train_file.readlines()
        x_train = np.array([[float(i) for i in line.strip().split(",")[:-1]] for line in lines])
        y_train = np.array([int(line.strip().split(",")[-1].replace(".", "")) - 1 for line in lines])
    with open("datasets/COIL100/test.csv") as train_file:
        lines = train_file.readlines()
        x_test = np.array([[float(i) for i in line.strip().split(",")[:-1]] for line in lines])
        y_test = np.array([int(line.strip().split(",")[-1].replace(".", "")) - 1 for line in lines])
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    data = [x_train, y_train, x_test, y_test]
    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


def test_madelon(default_settings):
    # http://archive.ics.uci.edu/ml/datasets/madelon
    # Existing training/test split
    num_classes = 2
    input_shape = (500,)
    default_settings['dataset'] = "Madelon"
    default_settings['is_image'] = False
    default_settings['hidden_layer_sizes'] = [1000, 1000, 1000]
    default_settings['learning_rate'] = 0.1
    settings = default_settings

    with open("datasets/madelon/madelon_train.data") as train_file:
        lines = train_file.readlines()
        x_train = np.array([line.split() for line in lines])
    with open("datasets/madelon/madelon_train.labels") as train_file:
        lines = train_file.readlines()
        y_train = np.array([0 if line.strip() == "-1" else 1 for line in lines])
    with open("datasets/madelon/madelon_valid.data") as valid_file:
        lines = valid_file.readlines()
        x_test = np.array([line.split() for line in lines])
    with open("datasets/madelon/madelon_valid.labels") as valid_file:
        lines = valid_file.readlines()
        y_test = np.array([0 if line.strip() == "-1" else 1 for line in lines])
    data = [x_train, y_train, x_test, y_test]

    # informative_features = {28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378, 433, 442, 451, 453, 455, 472, 475, 493}

    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


def test_epilepsy(default_settings):
    # https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
    # ~ 20% of data belongs to class 1, the rest to one of the other classes 2-5
    # Custom training/test split
    num_classes = 2
    input_shape = (178,)
    default_settings['dataset'] = "Epilepsy"
    default_settings['is_image'] = False
    default_settings['hidden_layer_sizes'] = [1000, 1000, 1000]
    settings = default_settings

    with open("datasets/epilepsy/train.csv") as train_file:
        lines = train_file.readlines()
        x_train = np.array([[float(i) for i in line.strip().split(",")[1:-1]] for line in lines])
        y_train = np.array([1 if line.strip().split(",")[-1] == "1" else 0 for line in lines])
    with open("datasets/epilepsy/test.csv") as train_file:
        lines = train_file.readlines()
        x_test = np.array([[float(i) for i in line.strip().split(",")[1:-1]] for line in lines])
        y_test = np.array([1 if line.strip().split(",")[-1] == "1" else 0 for line in lines])
    data = [x_train, y_train, x_test, y_test]
    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


def test_cnae(default_settings):
    # https://archive.ics.uci.edu/ml/datasets/CNAE-9
    # Custom training/test split
    num_classes = 9
    input_shape = (856,)
    default_settings['dataset'] = "CNAE-9"
    default_settings['is_image'] = False
    default_settings['hidden_layer_sizes'] = [1000, 1000, 1000]
    settings = default_settings

    with open("datasets/CNAE/train.csv") as train_file:
        lines = train_file.readlines()
        x_train = np.array([[float(i) for i in line.strip().split(",")[1:]] for line in lines])
        y_train = np.array([int(line.strip().split(",")[0]) - 1 for line in lines])
    with open("datasets/CNAE/test.csv") as train_file:
        lines = train_file.readlines()
        x_test = np.array([[float(i) for i in line.strip().split(",")[1:]] for line in lines])
        y_test = np.array([int(line.strip().split(",")[0]) - 1 for line in lines])
    data = [x_train, y_train, x_test, y_test]
    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


def test_micromass(default_settings):
    # Identifying micro organisms using pure spectra data
    # https://www.openml.org/d/1515
    # https://archive.ics.uci.edu/ml/datasets/MicroMass
    # Custom training/test split
    num_classes = 20
    input_shape = (1300,)
    default_settings['dataset'] = "MicroMass"
    default_settings['is_image'] = False
    default_settings['hidden_layer_sizes'] = [1000, 1000, 1000]
    default_settings['learning_rate'] = 0.1
    settings = default_settings

    with open("datasets/micromass/train.csv") as train_file:
        lines = train_file.readlines()
        x_train = np.array([[float(i) for i in line.strip().split(",")[:-1]] for line in lines])
        y_train = np.array([int(line.strip().split(",")[-1]) - 1 for line in lines])
    with open("datasets/micromass/test.csv") as train_file:
        lines = train_file.readlines()
        x_test = np.array([[float(i) for i in line.strip().split(",")[:-1]] for line in lines])
        y_test = np.array([int(line.strip().split(",")[-1]) - 1 for line in lines])
    data = [x_train, y_train, x_test, y_test]
    print("Dataset:", settings['dataset'])
    print("Settings:", settings)
    model = mlp(data, num_classes, input_shape, settings)
    np.savetxt(result_folder + "accuracies.txt", np.asarray(model.accuracies_case),
               header=json.dumps(settings, indent=4))


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    default_settings = {
        "typemlp": "evolutionary",
        "epsilon": 20,
        "learning_rate": 0.01,
        "use_centralities": False,
        "sim_sample_size": 0,
        "center_sim": False,
        "remove_using_sim": False,
        "add_using_sim": False,
        "randomly_add_using_sim": False,
        "remove_using_weighted_sim": False,
        "zeta": 0.3,
        "pref_attach_alpha": 0,
        "maxepoches": 100
    }
    default_settings['sim_sample_size'] = 1.0

    # Default settings runs the original SET-MLP model for benchmarking purposes.

    # CoDASET-MLP runs if you uncomment next line
    # default_settings['add_using_sim'] = True

    # CoPASET-MLP runs if you uncomment next line
    # default_settings['randomly_add_using_sim'] = True

    # CoRSET-MLP runs if you uncomment next line
    # default_settings['remove_using_weighted_sim'] = True

    # CoDACoRSET-MLP runs if you uncomment next line
    default_settings['add_using_sim'] = True
    default_settings['remove_using_weighted_sim'] = True

    # CoPACoRSET-MLP runs if you uncomment next line
    # default_settings['randomly_add_using_sim'] = True
    # default_settings['remove_using_weighted_sim'] = True

    # Other settings run different type of models which were used in the development phase for various purposes. Those models are not discussed in the paper as their results were not interesting.

    # We uploaded just the madelon dataset as it is small enough and has very interesting results. The other datasets can be easily downloaded from the Internet.
    test_madelon(default_settings)
