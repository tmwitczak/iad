import matplotlib.pyplot as plt
import csv
import sys
import os

data = []
with open(sys.argv[1]) as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        data.append(row)

plt.plot([float(i[0]) for i in data],
         [float(i[1]) for i in data])
plt.title('Funkcja kosztu dla ' + sys.argv[1])
plt.xlabel('Liczba epok')
plt.ylabel('Koszt')
plt.autoscale()
plt.savefig(sys.argv[1] + '.png',
            dpi=300, bbox_inches='tight')
plt.show()
