# /////////////////////////////////////////////////////////////////// Imports #
import csv
import statistics
from enum import Enum
from typing import Any, List, NamedTuple, Tuple

import numpy

# ////////////////////////////////////////////////////////////////// Typedefs #
Vector = numpy.ndarray


def vector_from_list(
        x: List[float]) \
        -> Vector:
    return numpy.array(x)


def empty_vector(
        length: int) \
        -> Vector:
    return numpy.empty(shape=length)


def zero_vector(
        length: int) \
        -> Vector:
    return numpy.zeros(shape=length)


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
def load_data_from_csv_file(
        csv_filename: str,
        class_labels_column_number: int,
        *,
        normalised: bool,
        standardised: bool) \
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
    # Clustering data attributes
    data: List[Vector] = []
    data_normalised: List[Vector] = []
    data_standardised: List[Vector] = []
    class_labels: List[str] = []
    outputs: List[Vector] = []

    # Load data and classes
    with open(csv_filename) as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            data.append(
                vector_from_list(
                    [float(x) for x in row[0:class_labels_column_number]]
                    + [float(x) for x in row[(class_labels_column_number
                                              + 1):len(row)]]))
            class_labels.append(row[class_labels_column_number])

    # Normalise and standardise data
    min_vector: Vector = empty_vector(len(data[0]))
    max_vector: Vector = empty_vector(len(data[0]))
    mean_vector: Vector = empty_vector(len(data[0]))
    stdev_vector: Vector = empty_vector(len(data[0]))

    for i in range(len(data[0])):
        min_vector[i] = min(get_column(data, i))
        max_vector[i] = max(get_column(data, i))
        mean_vector[i] = statistics.mean(get_column(data, i))
        stdev_vector[i] = statistics.stdev(get_column(data, i))

    for vector in data:
        if normalised:
            data_normalised.append(
                (vector - min_vector) / (max_vector - min_vector))
        if standardised:
            data_standardised.append(
                (vector - mean_vector) / stdev_vector)

    classes = dict.fromkeys(class_labels)
    number_of_outputs: int = len(classes)
    for i, class_label in enumerate(sorted(classes)):
        output_vector = zero_vector(number_of_outputs)
        output_vector[i] = 1.0
        classes[class_label] = output_vector.tolist()

    for class_label in class_labels:
        outputs.append(classes[class_label])

    # Return whole data set
    return ClassificationData(len(data[0]),
                              len(set(class_labels)),
                              tuple(data),
                              tuple(data_normalised),
                              tuple(data_standardised),
                              tuple(outputs),
                              tuple(class_labels))


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
    class_column: int = 0
    output_file: str
    normalise: bool = False
    normalised_file: str
    standardise: bool = False
    standardised_file: str

    try:
        opts, args = getopt.getopt(argv,
                                   "i:c:o:n:s:",
                                   ["input-file=", "class-column=",
                                    'output-file=',
                                    "normalised-file=", 'standardised-file=', ])

    except getopt.GetoptError:
        print('blad')  # TODO: Write better error message!
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-i", "--input-file"):
            input_file = arg
        elif opt in ("-c", "--class-column"):
            class_column = int(arg)
        elif opt in ('-o', '--output'):
            output_file = arg
        elif opt in ('-n', '--normalise'):
            normalise = True
            normalised_file = arg
        elif opt in ('-s', '--standardise'):
            standardise = True
            standardised_file = arg

    print(input_file)
    print(class_column)
    print(output_file)
    if normalise:
        print(normalised_file)
    if (standardise):
        print(standardised_file)

    # classification_data: ClassificationData = load_data_from_csv_file(
    #     '../mnist-digits/mnist-digits-test.csv', 0, normalised=False,
    #     standardised=False)
    classification_data: ClassificationData \
        = load_data_from_csv_file(input_file,
                                  class_column,
                                  normalised=normalise,
                                  standardised=standardise)

    write_classification_data_to_csv_file(
        classification_data.inputs,
        classification_data.outputs,
        classification_data.class_labels,
        output_file)

    if normalise:
        write_classification_data_to_csv_file(
            classification_data.inputs_normalised,
            classification_data.outputs,
            classification_data.class_labels,
            normalised_file)

    if standardise:
        write_classification_data_to_csv_file(
            classification_data.inputs_standardised,
            classification_data.outputs,
            classification_data.class_labels,
            standardised_file)

    #######################


def write_classification_data_to_csv_file(
        inputs: Tuple[Vector, ...],
        outputs: Tuple[Vector, ...],
        class_labels: Tuple[str, ...],
        csv_filename: str) \
        -> None:
    """"""
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_data = csv.writer(csv_file)

        csv_data.writerow(
            ['' for _ in range(len(inputs[0]))]
            + sorted(dict.fromkeys(class_labels)))

        for i in range(len(inputs)):
            csv_data.writerow(inputs[i].tolist() + outputs[i])


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == '__main__':
    main(sys.argv[1:])

# /////////////////////////////////////////////////////////////////////////// #
