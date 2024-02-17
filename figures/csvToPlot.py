import pathlib
import os
current = pathlib.Path().absolute()

import csv
import matplotlib.pyplot as plt

filename = "train_loss_godel"
title = "Gödel t-norm"
xlabel = "Elapsed time (s)"
ylabel = "Loss"

fields = []
rows = []

with open(os.path.join(current, 'figures',filename+'.csv'), 'r') as csvfile:
    csvreader  = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
    start = float(rows[0][0])
    t = [float(row[0])-start for row in rows]
    v = [float(row[2]) for row in rows]
    plt.figure()
    plt.plot(t, v)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()