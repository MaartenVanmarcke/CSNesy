import pathlib
import os
current = pathlib.Path().absolute()

import csv
import matplotlib.pyplot as plt
import numpy as np

classes = ["2",
           "3",
           "4",
           "5",
           "6", 
           "7",
           "8",
           "9",
           "10"]
times = {"A Tree per Batch":
         [ 
             .0410274,
             .199951,
             .559787,
             1.51803,
             3.27058,
             5.9791,
             11.6045,
             17.8103,
             26.5453
         ],
         "A Tree per Query":
         [
             2.3591346740722656,
             12.533123254776001,
             37.659581661224365,
             99.25937867164612,
             233.59144687652588,
             388.7754991054535,
             668.0005369186401,
             1101.4237625598907,
             1726.4412248134613
         ]}
title = "Running time of logic engine for different number of classes"
xlabel = "Number of Classes"
ylabel = "Elapsed time (s)"
savename = "figures\img\LogicTimeClasses.png"


plt.figure()
fig, ax = plt.subplots(layout='constrained')

xs =np.arange(len(classes))
width = .2
multiplier = 0

# code inspired by https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
for k, v in times.items():
    offset = width * multiplier
    rects = ax.bar(xs+offset, v, width, label = k)
    #ax.bar_label(rects, padding=3)
    multiplier +=1


plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
ax.set_xticks(xs + width, classes)
plt.legend()
plt.grid()
plt.savefig(savename)
plt.close()