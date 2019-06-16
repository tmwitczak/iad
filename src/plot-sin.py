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

dataActual = []
with open(sys.argv[1] + '.actual') as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        dataActual.append(row)

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
plt.title('Funkcja sin dla ' + sys.argv[1])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['sin', 'sieć'])
plt.autoscale()
plt.savefig(sys.argv[1] + '.png',
            dpi=300, bbox_inches='tight')
plt.show()
