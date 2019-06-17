import matplotlib.pyplot as plt
import csv
import sys

errors = []
with open(sys.argv[1]) as \
        csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        errors.append(float(*row))

plt.hist(errors, bins = 16)

plt.title('Histogram błędów na zbiorze testowym ' + sys.argv[1])
plt.xlabel('Błąd')
plt.ylabel('Ilość')
plt.autoscale()
plt.savefig(sys.argv[1] + '.png',
            dpi=300, bbox_inches='tight')
plt.show()
