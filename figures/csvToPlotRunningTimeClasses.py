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
           "7"]
times = [
    10.14,
    42.39,
    138.66,
    385.44,
    883.8,
    2170
]
title = "Training time for different number of classes"
xlabel = "Number of Classes"
ylabel = "Training time (s)"
savename = "figures\img\RunningTimeClasses.png"


fig, ax = plt.subplots(layout='constrained')

xs =np.arange(len(classes))
width = .2
multiplier = 0

# code inspired by https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
for i in range(len(times)):
    offset = 0#width * multiplier
    rects = ax.bar(xs+offset, times[i], width)
    #ax.set_yscale('log')
    #ax.bar_label(rects, padding=3)
    multiplier +=1


plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
ax.set_xticks(xs + width, classes)
plt.grid()
plt.show()
plt.savefig(savename)
plt.close()