import math
import matplotlib.pyplot as plt
import csv
import sys
import os

dataNet = []
with open(sys.argv[1] + '.net') as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        dataNet.append(row)

weightsNet = []
with open(sys.argv[1] + '.rbf-weights-net') as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        weightsNet.append(row)

dataActual = []
with open(sys.argv[1] + '.actual') as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        dataActual.append(row)

weights = []
with open(sys.argv[1] + '.rbf-weights') as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        weights.append(row)
        # print("w:", *row)

biases = []
with open(sys.argv[1] + '.rbf-biases') as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        biases.append(row)
        # print("b: ", *row)

errors = []
for i in range(len(dataActual)):
    errors.append((float(dataActual[i][1]) - float(dataNet[i][1]))**2)

sortedErrors = errors.copy()
sortedErrors.sort(reverse = True)

def is_error_that_bad(error):
    return error in sortedErrors[0:int(0.05*len(errors))]

marker_on = []
for i in range(len(dataActual)):
    if is_error_that_bad(errors[i]):
        marker_on.append(i)


plt.plot([float(i[0]) for i in dataActual],
         [float(i[1]) for i in dataActual], 'c-')
plt.plot([float(i[0]) for i in dataNet],
         [float(i[1]) for i in dataNet], '-rx',
         markevery=marker_on)
plt.title('Funkcja ' + sys.argv[2] + ' dla ' + sys.argv[1])
plt.xlabel('x')
plt.ylabel('y')
plt.legend([sys.argv[2], 'sieć'])
plt.autoscale()

a = plt.gcf().gca()
for i in range(len(weights)):
    # print(weights[i])
    # print(weightsNet[i])
    # print()
    a.add_artist(plt.Circle((float(weights[i][0]), float(weightsNet[i][0])),
                            0.05 / math.sqrt(float(biases[i][0])), color='b',
                            fill=False))

plt.savefig(sys.argv[1] + '.png',
            dpi=300, bbox_inches='tight')
plt.show()
