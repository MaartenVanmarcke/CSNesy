import pathlib
import os
current = pathlib.Path().absolute()

import csv
import matplotlib.pyplot as plt

filename = ["train_loss_semiring", "Sum Product Semiring",
            "train_loss_godel", "Gödel t-norm",
            "train_loss_product", "Product t-norm",
            "train_loss_lukasievicz", "Łukasiewicz t-norm"]
title = "Training loss for different semantics"
xlabel = "Elapsed time (s)"
ylabel = "Training loss"
savename = "img\TrainLossSemantics.png"


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
plt.show()
plt.savefig(savename)


plt.figure()
plt.plot
with open(os.path.join(current, 'figures',filename+'.csv'), 'r') as csvfile1:
    csvreader1  = csv.reader(csvfile1)
    with open(os.path.join(current, 'figures',filename+'.csv'), 'r') as csvfile2:
        csvreader2  = csv.reader(csvfile2)
        with open(os.path.join(current, 'figures',filename+'.csv'), 'r') as csvfile3:
            csvreader3  = csv.reader(csvfile3)
            with open(os.path.join(current, 'figures',filename+'.csv'), 'r') as csvfile4:
                csvreader4  = csv.reader(csvfile4)
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