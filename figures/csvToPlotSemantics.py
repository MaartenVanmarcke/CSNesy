import pathlib
import os
current = pathlib.Path().absolute()

import csv
import matplotlib.pyplot as plt

filename = ["train_loss_semiring", "Sum Product Semiring",
            "train_loss_godel", "Gödel t-norm",
            "train_loss_product", "Product t-norm"]
            #"train_loss_lukasievicz", "Łukasiewicz t-norm"]
title = "Training loss for different semantics"
xlabel = "Elapsed time (s)"
ylabel = "Training loss"
savename = "figures\img\TrainLossSemanticsWithoutLuka.png"


plt.figure()

for i in range(int(len(filename)/2)):
    with open(os.path.join(current, 'figures',filename[2*i]+'.csv'), 'r') as csvfile:
        fields = []
        rows = []
        csvreader  = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
        start = float(rows[0][0])
        t = [float(row[0])-start for row in rows]
        v = [float(row[2]) for row in rows]
        print("v", v)
        plt.plot(t, v, label = filename[2*i+1])


plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()
plt.grid()
plt.savefig(savename)
plt.show()
plt.close()