import math

import numpy as np

from neupy.utils import format_data

from neupy.core.properties import Property

from .utils import bin2sign, hopfield_energy, step_function

from .base import DiscreteMemory

__all__ = ('DiscreteHopfieldNetwork',)


class DiscreteHopfieldNetwork(DiscreteMemory):

    check_limit = Property(default=True, expected_type=bool)



    def __init__(self, **options):

        super(DiscreteHopfieldNetwork, self).__init__(**options)

        self.n_memorized_samples = 0



    def train(self, input_data):

        self.discrete_validation(input_data)



        input_data = bin2sign(input_data)

        input_data = format_data(input_data, is_feature1d=False)



        n_rows, n_features = input_data.shape

        n_rows_after_update = self.n_memorized_samples + n_rows



        if self.check_limit:

            memory_limit = math.ceil(n_features / (2 * math.log(n_features)))



            if n_rows_after_update > memory_limit:

                raise ValueError("You can't memorize more than {0} "

                                 "samples".format(memory_limit))



        weight_shape = (n_features, n_features)



        if self.weight is None:

            self.weight = np.zeros(weight_shape, dtype=int)



        if self.weight.shape != weight_shape:

            n_features_expected = self.weight.shape[1]

            raise ValueError("Input data has invalid number of features. "

                             "Got {} features instead of {}."

                             "".format(n_features, n_features_expected))



        self.weight = input_data.T.dot(input_data)

        np.fill_diagonal(self.weight, np.zeros(len(self.weight)))

        self.n_memorized_samples = n_rows_after_update



    def predict(self, input_data, n_times=None):

        self.discrete_validation(input_data)

        input_data = format_data(bin2sign(input_data), is_feature1d=False)



        if self.mode == 'async':

            if n_times is None:

                n_times = self.n_times



            _, n_features = input_data.shape

            output_data = input_data



            for _ in range(n_times):

                position = np.random.randint(0, n_features - 1)

                raw_new_value = output_data.dot(self.weight[:, position])

                output_data[:, position] = np.sign(raw_new_value)

        else:

            output_data = input_data.dot(self.weight)



        return step_function(output_data).astype(int)



    def energy(self, input_data):

        self.discrete_validation(input_data)

        input_data = bin2sign(input_data)

        input_data = format_data(input_data, is_feature1d=False)

        n_rows, n_features = input_data.shape



        if n_rows == 1:

            return hopfield_energy(self.weight, input_data, input_data)



        output = np.zeros(n_rows)

        for i, row in enumerate(input_data):

            output[i] = hopfield_energy(self.weight, row, row)



        return output