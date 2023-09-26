import copy
import threading
import multiprocessing
from assistant import Assistant
from nnencoding_operations import NNEncodingOps
import numpy as np
from PIL import Image
import math
from nn_encoding import Link
from nn_encoding import Node
from nn_encoding import NNEncoding
from pioneer import NNPioneer
import os
import time
import random


class ui:

    def __init__(self, img_size):
        self.img_size = img_size
        self.pixel_output = np.zeros([img_size, img_size, 3], dtype=np.uint8)
        self.input_array = np.zeros([3, 1])
        self.blueprint_data = Assistant.read_json(
            '/home/tayyba/PycharmProjects/ImageEvo-3Aug/ImageEvolution/default_blueprint.json')

        # for multi-threading mutation
        # self.selected_nn_list = []
        # self.generation_list = []

        self.compute_input_values()


    def compute_input_values(self):
        x_list = []
        y_list = []
        z_list = []
        for i in range(self.pixel_output.shape[0]):
            for j in range(self.pixel_output.shape[1]):
                z = random.uniform(0, 1)
                scale = 50
                factor = min(self.img_size, self.img_size) / scale
                x = i / factor - 0.5 * scale
                y = j / factor - 0.5 * scale
                z = math.sqrt((x * x + y * y))

                x_list.append(x)
                y_list.append(y)
                z_list.append(z)

        self.input_array = np.array([x_list, y_list, z_list])

    def create_n_save(self, num_image):
        print(multiprocessing.current_process())

        n1 = NNEncodingOps(self.blueprint_data)

        start_time = time.time()
        Link.id = 0
        Node.id = 0
        nn = n1.generate_encoding()
        nn.save_model('models/nn_' + str(num_image))

        pixel_output = np.zeros([self.img_size, self.img_size, 3], dtype=np.uint8)

        pixel_matrix = np.array(nn.get_output_result(self.input_array))

        pixel_matrix[0] = (np.round((pixel_matrix[0] + 1.0) * 255 / 2.0)).astype(int)  # red
        pixel_matrix[1] = (np.round((pixel_matrix[1] + 1.0) * 255 / 2.0)).astype(int)  # blue
        pixel_matrix[2] = (np.round((pixel_matrix[2] + 1.0) * 255 / 2.0)).astype(int)  # blue

        pixel_matrix = pixel_matrix.T

        pixel_image = np.reshape(pixel_matrix, (self.img_size, self.img_size, 3))

        image2 = Image.fromarray(pixel_image.astype(np.uint8), 'RGB')
        image2.save('generated_images/image' + str(num_image) + '.png')

        end = time.time()

        print('total time for computing id: ' + str(num_image) +' image= ' + str(end - start_time))

    def initialize(self, num_individual=1):

        if num_individual < 10:
            num_image = np.array([i for i in range(num_individual)])
            p = multiprocessing.Pool(processes=num_individual)

            p.map(self.create_n_save, num_image)
            p.close()
            p.join()
        else:
            print('Minimum number of images to be generated should be less than 10')

    def next_gen(self, selected_nn, generation):
        nn_file = selected_nn
        nn_parent = NNEncoding({'filename': nn_file})
        nn_child = copy.deepcopy(nn_parent)


        encoding_ops = NNEncodingOps(self.blueprint_data)

        nn_mutated = NNPioneer(encoding_ops)
        """~~~~~~~~~~~~~~~~~~~~~MUTATION~~~~~~~~~~~~~~~~~~~~~"""
        nn_child = nn_mutated.node_add_mutation(nn_parent, nn_child)

        nn_child.save_model('models/nn_' + str(generation))

        pixel_output = np.zeros([self.img_size, self.img_size, 3], dtype=np.uint8)

        pixel_matrix = np.array(nn_child.get_output_result(self.input_array))

        pixel_matrix[0] = (np.round((pixel_matrix[0] + 1.0) * 255 / 2.0)).astype(int)  # red
        pixel_matrix[1] = (np.round((pixel_matrix[1] + 1.0) * 255 / 2.0)).astype(int)  # blue
        pixel_matrix[2] = (np.round((pixel_matrix[2] + 1.0) * 255 / 2.0)).astype(int)  # blue

        pixel_matrix = pixel_matrix.T

        pixel_image = np.reshape(pixel_matrix, (self.img_size, self.img_size, 3))

        image2 = Image.fromarray(pixel_image.astype(np.uint8), 'RGB')
        image2.save('generated_images/image' + str(generation) + '.png')

        return
    # multi-processing
    # def next_gen(self, iterator_list):
    #     nn_file = self.selected_nn_list[iterator_list]
    #     nn_parent = NNEncoding({'filename': nn_file})
    #     nn_child = copy.deepcopy(nn_parent)
    #
    #
    #     encoding_ops = NNEncodingOps(self.blueprint_data)
    #
    #     nn_mutated = NNPioneer(encoding_ops)
    #     """~~~~~~~~~~~~~~~~~~~~~MUTATION~~~~~~~~~~~~~~~~~~~~~"""
    #     nn_child = nn_mutated.node_add_mutation(nn_parent, nn_child)
    #
    #     nn_child.save_model('models/nn_' + str(self.generation_list[iterator_list]))
    #
    #     pixel_output = np.zeros([self.img_size, self.img_size, 3], dtype=np.uint8)
    #
    #     pixel_matrix = np.array(nn_child.get_output_result(self.input_array))
    #
    #     pixel_matrix[0] = (np.round((pixel_matrix[0] + 1.0) * 255 / 2.0)).astype(int)  # red
    #     pixel_matrix[1] = (np.round((pixel_matrix[1] + 1.0) * 255 / 2.0)).astype(int)  # blue
    #     pixel_matrix[2] = (np.round((pixel_matrix[2] + 1.0) * 255 / 2.0)).astype(int)  # blue
    #
    #     pixel_matrix = pixel_matrix.T
    #
    #     pixel_image = np.reshape(pixel_matrix, (self.img_size, self.img_size, 3))
    #
    #     image2 = Image.fromarray(pixel_image.astype(np.uint8), 'RGB')
    #     image2.save('generated_images/image' + str(self.generation_list[iterator_list]) + '.png')


    def mutation(self, directory,selected_img, num_mutation):

        for generation in range(selected_img, selected_img + num_mutation):
            file_name = directory+ str(generation) + '.json'
            selected_nn = file_name
            self.next_gen(selected_nn, generation)

        # for MULTI-PROCESSING
        # iterator_list = []
        # i = 0
        # for generation in range(selected_img, selected_img + num_mutation):
        #     file_name = directory+ str(generation) + '.json'
        #     print("working for generation:", generation)
        #     print('filename', file_name)
        #
        #     selected_nn = file_name
        #
        #     self.selected_nn_list.append(selected_nn)
        #     self.generation_list.append(generation+1)
        #     iterator_list.append(i)
        #     i = i + 1
        # # multiprocessing here
        # p = multiprocessing.Pool(processes=num_mutation)
        #
        # p.map(self.next_gen, iterator_list)
        # p.close()
        # p.join()




def main():
    # first generation
    test_ui = ui(1000)
    test_ui.initialize(9)

    # mutation
    # directory = '/home/tayyba/PycharmProjects/ImageEvo-3Aug/ImageEvolution/models/nn_'
    # num_mutation = 5
    # selected_img_num = 4
    # test_ui.mutation(directory, selected_img_num, num_mutation)
    return


if __name__ == '__main__':
    main()
