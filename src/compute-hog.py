import csv

from skimage import feature
import numpy
import cv2

# digit_image = numpy.array(cv2.imread('digit.png'))
# feature_vector = feature.hog(digit_image,
#                              orientations = 8,
#                              pixels_per_cell = (7, 7),
#                              cells_per_block = (2, 2),
#                              transform_sqrt = True,
#                              feature_vector = True)

# ---------------------------------------------------

# /////////////////////////////////////////////////////////////////// Imports #
import csv
import os
import statistics
from enum import Enum
from typing import Any, List, NamedTuple, Tuple

import numpy
import random


def get_random_shuffled_range(
        n: int) \
        -> List[int]:
    return random.sample(range(n), k = n)


# ////////////////////////////////////////////////////////////////// Typedefs #
Vector = numpy.ndarray


def vector_from_list(
        x: List[float]) \
        -> Vector:
    return numpy.array(x)


def empty_vector(
        length: int) \
        -> Vector:
    return numpy.empty(shape = length)


def zero_vector(
        length: int) \
        -> Vector:
    return numpy.zeros(shape = length)


# /////////////////////////////////////////////////////////////////// Classes #
# class DataMode(Enum):
#     DEFAULT = 0
#     NORMALISED = 1
#     STANDARDISED = 2
#
#
class ClassificationData(NamedTuple):
    number_of_inputs: int
    number_of_outputs: int
    inputs: Tuple[Vector, ...]
    inputs_normalised: Tuple[Vector, ...]
    inputs_standardised: Tuple[Vector, ...]
    outputs: Tuple[Vector, ...]
    class_labels: Tuple[str, ...]


# /////////////////////////////////////////////////////////////////////////// #
def print_status_bar(
        title: str,
        a: int,
        b: int,
        n: int = 10) \
        -> None:
    print('\r', title, ': |', '=' * int((a / b) * n),
          '-' if a != b else '',
          ' ' * (n - int((a / b) * n) - 1),
          '| (', a, ' / ', b, ')',
          sep = '', end = '', flush = True)


# /////////////////////////////////////////////////////////////////////////// #
def load_data_from_csv_file(
        csv_filename: str,
        class_labels_column_number: int,
        *,
        normalised: bool,
        standardised: bool,
        normalise_min: int,
        normalise_max: int) \
        -> ClassificationData:
    """ Load data from csv file and separate numeric values from classes.
    Parameters
    ----------
    csv_filename : str
        Path to csv file.
    class_labels_column_number : int
        Number of column containing class identifier.
    Returns
    -------
    ClusteringData
        Set of data for further clustering.
    """
    # ----------------------------------------------------------------------- #


# /////////////////////////////////////////////////////////////////////////// #
def get_column(
        two_dimensional_list: List[List[Any]],
        n: int) \
        -> List:
    """ Return n-th column of two dimensional list.
    Parameters
    ----------
    two_dimensional_list : List[List[Any]]
        Two dimensional list of which the column should be returned.
    n : int
        Number of column to return.
    Returns
    -------
    List
        N-th column of provided two dimensional list.
    """
    return [row[n] for row in two_dimensional_list]


# ////////////////////////////////////////////////////////////////////// Main #
import sys
import getopt


def main(
        argv: List[Any]) \
        -> None:
    """
    """
    # ----------------------------------------------------------------------- #
    # Parse program's arguments
    input_file: str
    class_column: int
    output_file: str
    orientations: int

    try:
        opts, args = getopt.getopt(argv,
                                   "i:c:o:b:",
                                   ["input-file=",
                                    "class-column=",
                                    'output-file=',
                                    'orientations='])
    except getopt.GetoptError:
        print('Błąd opcji')  # TODO: Write better error message!
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-i', "--input-file"):
            input_file = arg
        elif opt in ('-c', "--class-column"):
            class_column = int(arg)
        elif opt in ('-o', '--output-file'):
            output_file = arg
        elif opt in ('-b', '--orientations'):
            orientations = int(arg)

    # Load data from .csv
    data: List[Vector] = []
    class_labels: List[str] = []
    outputs: List[Vector] = []
    data_hog: List[Vector] = []

    with open(input_file) as csv_file:
        csv_data = csv.reader(csv_file)
        for i, row in enumerate(csv_data):
            print('Loading ' + input_file + ' | Row: ', i,
                  sep = '', end = '\r', flush = True)
            data.append(
                vector_from_list(
                    [float(x) for x in row[0:class_column]]
                    + [float(x) for x in row[(class_column
                                              + 1):len(row)]]))
            class_labels.append(row[class_column])
        else:
            print(' ' * len('Loading ' + input_file + ' | Row: ' + str(i)),
                  sep = '', end = '\r', flush = True)

    classes = dict.fromkeys(class_labels)
    number_of_outputs: int = len(classes)
    for i, class_label in enumerate(sorted(classes)):
        output_vector = zero_vector(number_of_outputs)
        output_vector[i] = 1.0
        classes[class_label] = output_vector.tolist()

    for class_label in class_labels:
        outputs.append(classes[class_label])

    # Compute HOG descriptors
    for vector in data:
        vector = vector.reshape(28, 28)
        feature_vector = feature.hog(vector,
                                     orientations = orientations,
                                     pixels_per_cell = (7, 7),
                                     cells_per_block = (2, 2),
                                     transform_sqrt = True,
                                     feature_vector = True)
        feature_vector = feature_vector.flatten()
        data_hog.append(feature_vector)

    # Save HOGs
    write_classification_data_to_csv_file(
        data_hog,
        outputs,
        class_labels,
        output_file)


def write_to_file(training_indices, testing_indices,
                  inputs, outputs, class_labels,
                  output_file):
    training_inputs = []
    training_outputs = []
    training_class_labels = []
    for i in training_indices:
        training_inputs.append(inputs[i])
        training_outputs.append(outputs[i])
        training_class_labels.append(class_labels[i])

    testing_inputs = []
    testing_outputs = []
    testing_class_labels = []
    for i in testing_indices:
        testing_inputs.append(inputs[i])
        testing_outputs.append(outputs[i])
        testing_class_labels.append(class_labels[i])

    write_classification_data_to_csv_file(
        training_inputs,
        training_outputs,
        training_class_labels,
        os.path.splitext(output_file)[0]
        + '-train'
        + os.path.splitext(output_file)[1])

    write_classification_data_to_csv_file(
        testing_inputs,
        testing_outputs,
        testing_class_labels,
        os.path.splitext(output_file)[0]
        + '-test'
        + os.path.splitext(output_file)[1])


def write_classification_data_to_csv_file(
        inputs: Tuple[Vector, ...],
        outputs: Tuple[Vector, ...],
        class_labels: Tuple[str, ...],
        csv_filename: str) \
        -> None:
    """"""
    with open(csv_filename, mode = 'w', newline = '') as csv_file:
        csv_data = csv.writer(csv_file)

        csv_data.writerow(
            [' ' for _ in range(len(inputs[0]))]
            + sorted(dict.fromkeys(class_labels)))

        for i in range(len(inputs)):
            csv_data.writerow(inputs[i].tolist() + outputs[i])


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == '__main__':
    main(sys.argv[1:])

# /////////////////////////////////////////////////////////////////////////// #
