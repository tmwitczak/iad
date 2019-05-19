import matplotlib.pyplot as plt
import csv

data = []
with open('cmake-build-release/training-result-error') as csv_file:
    csv_data = csv.reader(csv_file)
    for row in csv_data:
        data.append(row)

plt.plot([float(i[0]) for i in data], [float(i[1]) for i in data])
plt.show()
