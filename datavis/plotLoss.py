import sys
import numpy as np
sys.path.append('../nn')
from datetime import datetime
import torch
from mynn import *
from mydata import *
import random

import matplotlib as mpl 
from matplotlib.lines import Line2D 
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

coloroption="paper"

plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
plt.rc("text", usetex=True)         # Crisp axis ticks
plt.rc("font", family="serif")      # Crisp axis labels

#plt.rc("legend", edgecolor='none')  # No boxes around legends

if coloroption == "paper":
    plt.rc("figure",facecolor="#ffffff")
    plt.rc("axes",facecolor="#ffffff",edgecolor="#808080",labelcolor="#000000")
    plt.rc("savefig",facecolor="#ffffff")
    plt.rc("text",color="#000000")
    plt.rc("xtick",color="#808080")
    plt.rc("ytick",color="#808080")
    lbl = "#000000"
    tk = "#808080"
elif coloroption == "darkslides":
    plt.rc("figure",facecolor="#353F4F")
    plt.rc("axes",facecolor="#353F4F",edgecolor="#E7E6E6",labelcolor="#E7E6E6")
    plt.rc("savefig",facecolor="#353F4F")
    plt.rc("text",color="#E7E6E6")
    plt.rc("xtick",color="#E7E6E6")
    plt.rc("ytick",color="#E7E6E6")
    lbl = "#E7E6E6"
    tk = "#E7E6E6"
plt.yscale('log')
for i in range(5):
    
    loss = torch.load("../nn/PCANet_"+str(i)+".loss")

    plt.scatter(np.arange(len(loss)), loss/5000, s=1, label="Run: "+str(i), c="C"+str(i))
    
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("NS-pca-vis-loss.pdf", bbox_inches='tight')