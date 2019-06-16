import matplotlib.pyplot as plt
import csv
import sys
import os

def doStuff(number):
    dataNet = []
    with open(sys.argv[1] + '.net' + str(number)) as \
            csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            dataNet.append(row)

    dataActual = []
    with open(sys.argv[1] + '.actual' + str(number)) as \
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
    title = ""
    if number == 1:
        title = 'x2=0'
    elif number == 2:
        title = 'x1=0'
    plt.title('Funkcja sin(x1*x2)+cos(3(x1-x2)) [' + title + '] dla ' + sys.argv[1])
    if number == 1:
        plt.xlabel('x1')
    elif number == 2:
        plt.xlabel('x2')
    plt.ylabel('y')
    plt.legend(['sin(x1*x2)+cos(3(x1-x2))', 'sieć'])
    plt.autoscale()
    plt.savefig(sys.argv[1] + str(number) + '.png',
                dpi=300, bbox_inches='tight')
    plt.show()

doStuff(1)
doStuff(2)