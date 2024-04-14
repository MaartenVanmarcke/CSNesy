import pathlib
import os
current = pathlib.Path().absolute()

import csv
import matplotlib.pyplot as plt
import numpy as np

classes = ["(2,2)",
           "(2,3)",
           "(2,4)",
           "(2,5)",
           "(2,6)",
           "(2,7)",
           "(3,2)", 
           "(3,3)"]
times = [
    [  3.6803269386291504,
    9.116119146347046,
    15.956370830535889,
    42.46424627304077,
    140.4936068058014,
    402.9000585079193,
    23.10112428665161,
    541.6563320159912],
    [0.14037394523620605,
    1.2582225799560547,
    6.494197607040405,
    29.563303470611572,
    114.60446190834045,
    366.90204787254333,
    18.960294723510742,
    535.5727488994598]
]
title = "Execution time for different BaseConverter task instances"
xlabel = "(Number of Digits, Number of Classes)"
ylabel = "Training time (s)"
savename = "figures\img\RunningTimeClassesBC.png"


fig, ax = plt.subplots(layout='constrained')

xs =np.arange(len(classes))
width = .2
multiplier = 0

# code inspired by https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
offset = width
multiplier = 0
for i in range(2):
    rects = ax.bar(xs+offset*multiplier+width/2, times[i], width)
    #ax.set_yscale('log')
    #ax.bar_label(rects, padding=3)
    multiplier += 1
plt.ylim([0,600])

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
ax.set_xticks(xs + width, classes)
plt.grid()
plt.legend(["Training time", "Execution time of logic engine"])
plt.savefig(savename)
plt.show()
plt.close()