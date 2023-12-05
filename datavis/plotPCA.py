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

coloroption="darkslides"

plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
plt.rc("text", usetex=True)         # Crisp axis ticks
plt.rc("font", family="serif")      # Crisp axis labels
plt.rc("legend", edgecolor='none')  # No boxes around legends

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

M = 10000

N = 64

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

fig = plt.figure(figsize=(10,10))
subfigs = fig.subfigures(nrows=4,ncols=1)
axs = []
for row,subfig in enumerate(subfigs):
    axs.append(subfig.subplots(1,5))
ims = []

offset=2

ntrain = int(M*.8)
ntest = M - ntrain
N_theta = 100
prefix = "../datagen/"
acc= 0.999
i = 0
input_truth = []
output_truth = []
output_pred = []
output_err= []
mre = []

prefix = "../datagen/"
v = np.load(prefix+"viscosity_" + str(M) + ".npy")
theta = np.load(prefix+"theta_" + str(M) + ".npy")
inputs = np.load(prefix+"curl_f_"  + str(M) + ".npy") 
outputs = np.load(prefix+"omega_"  + str(M) + ".npy")
model = torch.load("../nn/PCANet_200Nd.model")

train_inputs = np.reshape(inputs[:,:,:ntrain], (-1, ntrain))
test_inputs  = np.reshape(inputs[:,:,ntrain:M], (-1, ntest))
v_train = v[:ntrain]
v_test  = v[ntrain:M]
Ui,Si,Vi = np.linalg.svd(train_inputs)
en_f= 1 - np.cumsum(Si)/np.sum(Si)
r_f = np.argwhere(en_f<(1-acc))[0,0]

# r_f = min(r_f, 512)
r_f = 128

Uf = Ui[:,:r_f]
f_hat = np.matmul(Uf.T,train_inputs)
f_hat_test = np.matmul(Uf.T,test_inputs)

x_train = torch.from_numpy(f_hat.T.astype(np.float32))
train_outputs = np.reshape(outputs[:,:,:ntrain], (-1, ntrain))
test_outputs  = np.reshape(outputs[:,:,ntrain:M], (-1, ntest))
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)
r_g = np.argwhere(en_g<(1-acc))[0,0]
Ug = Uo[:,:r_g]
g_hat = np.matmul(Ug.T,train_outputs) 
y_train = torch.from_numpy(g_hat.T.astype(np.float32))
model.to(device)
        
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

if torch.cuda.is_available():
    x_normalizer.cuda()
    y_normalizer.cuda()
x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))
x_test = x_normalizer.encode(x_test.to(device))
x_test = torch.from_numpy(np.hstack((x_test.cpu(), np.reshape(v_test, (ntest, -1))))).float()
x_test = x_test.to(device)

y_pred_test  = y_normalizer.decode(model(x_test).detach()).cpu().numpy().T
for i in range(5):
    i = random.randint(0,ntest-1)
    xgrid = np.linspace(0,1,N)
    Y, X = np.meshgrid(xgrid, xgrid)
    output_pred.append(np.reshape(np.matmul(Ug, y_pred_test[:,i]), (N,N)))
    input_truth.append(np.reshape(test_inputs[:,i], (N,N)))
    output_truth.append(np.reshape(test_outputs[:,i], (N,N)))
    output_err.append(np.reshape(test_outputs[:,i], (N,N))-np.reshape(np.matmul(Ug, y_pred_test[:,i]), (N,N)))
    mre.append(np.linalg.norm(np.reshape(test_outputs[:,i] - np.matmul(Ug, y_pred_test[:,i]), (N,N)))/np.linalg.norm(np.reshape(test_outputs[:,i], (N,N))))
        
v_min = min(np.min(output_truth),np.min(output_pred))
v_max = max(np.max(output_truth),np.max(output_pred))
err_min = np.min(output_err)
err_max = np.max(output_err)
for i in range(5):
        ims.append(axs[0][i].pcolormesh(X,Y,input_truth[i],shading="gouraud",cmap="RdBu"))
        ims.append(axs[1][i].pcolormesh(X,Y,output_truth[i],shading="gouraud",cmap="RdBu",vmin=v_min,vmax=v_max))
        ims.append(axs[2][i].pcolormesh(X,Y,output_pred[i],shading="gouraud",cmap="RdBu",vmin=v_min,vmax=v_max))
        ims.append(axs[3][i].pcolormesh(X,Y,output_err[i],shading="gouraud",cmap="RdBu", vmin=err_min,vmax=err_max))
        axs[3][i].set_label(str(mre[i]))
        
        for j in range(4):
            axs[j][i].set_aspect("equal","box")
            axs[j][i].set_axis_off()
        i+=1
    
subfigs[0].suptitle("Input",fontsize=16,y=0.95)
subfigs[2].suptitle("Prediction",fontsize=16,y=0.95)
subfigs[1].suptitle("\"Truth\"",fontsize=16,y=0.95)
subfigs[3].suptitle("Error",fontsize=16,y=0.95)

plt.subplots_adjust(left=0.02,right=0.87,bottom=0.02,top=0.90)
cax = []
for i in range(4):
    temp = axs[i][4].get_position()
    cax.append(subfigs[i].add_axes([0.89,temp.y0,0.02,temp.y1-temp.y0]))
    if i==0:
        cb = plt.colorbar(ims[i],cax=cax[i],ticks=[np.min(test_inputs),0,np.max(test_inputs)])
    elif i == 1:
        cb = plt.colorbar(ims[i],cax=cax[i],ticks=[v_min,0,v_max])
    elif i == 2:
        cb = plt.colorbar(ims[i],cax=cax[i],ticks=[v_min,0,v_max])
    elif i == 3:
        cb = plt.colorbar(ims[i],cax=cax[i],ticks=[err_min,0,err_max])
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_tick_params(width=0.3)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)


fig.savefig("NS-pca-vis-"+coloroption+".pdf")
plt.close("all")
# plt.colorbar()
