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
plt.rc("font", family="serif", size=12)      # Crisp axis labels

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

M = 5000

offset=2

ntrain = M
ntest = 1000
N_theta = 100
prefix = "../datagen/"
acc= 0.9999
i = 0
print("Loading data")
prefix = "../datagen/"
v = np.load(prefix+"viscosity_" + str(M) + ".npy")
theta = np.load(prefix+"theta_" + str(M) + ".npy")
inputs = np.load(prefix+"curl_f_"  + str(M) + ".npy") 
outputs = np.load(prefix+"omega_"  + str(M) + ".npy")


v_test = np.load(prefix+"viscosity_" + str(ntest) + "_test.npy")
theta_test = np.load(prefix+"theta_" + str(ntest) + "_test.npy")
inputs_test = np.load(prefix+"curl_f_"  + str(ntest) + "_test.npy") 
outputs_test = np.load(prefix+"omega_"  + str(ntest) + "_test.npy")
print("Data loaded")
models = []
for i in range(5):
    models.append(torch.load("../nn/PCANet_"+str(i)+".model"))
print("Models loaded")

compute_pca = False


train_inputs = np.reshape(inputs, (-1, M))
v_train = v
train_outputs = np.reshape(outputs, (-1, M))

test_inputs = np.reshape(inputs_test, (-1, ntest))
test_outputs = np.reshape(outputs_test, (-1, ntest))


try:
    Uf = np.load("../nn/PCANet_"+str(M)+"_Uf.npy")
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    r_f = Uf.shape[1]

    Ug = np.load("../nn/PCANet_"+str(M)+"_Ug.npy")
    g_hat = np.matmul(Ug.T,train_outputs)
    y_train = torch.from_numpy(g_hat.T.astype(np.float32))
    r_g = Ug.shape[1]
    print("PCA loaded")
except:
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    # r_f = min(r_f, 512)
    # print(Si[98:210])
    #r_f = 128
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    np.save("../nn/PCANet_"+str(M)+"_Uf.npy", Uf)

    Uo,So,Vo = np.linalg.svd(train_outputs)
    en_g = 1 - np.cumsum(So)/np.sum(So)
    r_g = np.argwhere(en_g<(1-acc))[0,0]
    Ug = Uo[:,:r_g]
    g_hat = np.matmul(Ug.T,train_outputs)
    y_train = torch.from_numpy(g_hat.T.astype(np.float32))
    np.save("../nn/PCANet_"+str(M)+"_Ug.npy", Ug)
    
    print("PCA computed")
    
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

if True:
    
        
    N = 100

    xgrid = np.linspace(0,1,N)
    dx    = xgrid[1] - xgrid[0]

    fig = plt.figure(figsize=(8,12))
    subfigs = fig.subfigures(nrows=     4,ncols=1)
    axs = []
    for row,subfig in enumerate(subfigs):
        axs.append(subfig.subplots(1,3))
    ims = []
        
        
    input_truth = []
    output_truth = []
    output_pred = []
    output_err= []
    mre = []
    vs = []

    models[0].to(device)
    y_pred_test = y_normalizer.decode(models[0](x_test).detach()).cpu().numpy().T    
    rel_err_nn_test = np.zeros(ntest)
    for i in range(ntest):
            rel_err_nn_test[i] = np.linalg.norm(test_outputs[:, i]  - np.matmul(Ug, y_pred_test[:, i]))/np.linalg.norm(test_outputs[:, i])
       
    for j in range(3):
        
        if j == 0:
            min_err = min(rel_err_nn_test[v_test<1])
            print("Min_err: ", min_err)
            i = np.where(rel_err_nn_test == min_err)[0]
            vs.append(v_test[i][0])
        elif j == 1: 
                
            rel_err_nn_test_fixed = []
            v_test_fixed = []
            for i in range(ntest):
                if v_test[i] <= 1:
                    rel_err_nn_test_fixed.append(rel_err_nn_test[i])
                    v_test_fixed.append(v_test[i])
            if len(rel_err_nn_test_fixed) % 2 == 0:
                rel_err_nn_test_fixed.append(0)
            med_err= np.median(rel_err_nn_test_fixed)
            print("Median err: ", med_err)
            i = np.where(rel_err_nn_test == med_err)[0]
            vs.append(v_test[i][0])
        else: 
            max_err = max(rel_err_nn_test[v_test<1])
            print("Max err: ", max_err)
            i = np.where(rel_err_nn_test == max_err)[0]
            vs.append(v_test[i][0])
        print("Viscosity: ", v_test[i][0])
        print("Index: ", i)
        xgrid = np.linspace(0,1,N)
        Y, X = np.meshgrid(xgrid, xgrid)
        input_truth.append(np.reshape(test_inputs[:,i], (N,N)))
        output_corr = (np.max(test_outputs[:,i])-np.min(test_outputs[:,i]))/2
        print("Corr: ", output_corr)
        output_truth.append(np.reshape(test_outputs[:,i], (N,N))/output_corr)
        output_pred.append(np.reshape(np.matmul(Ug, y_pred_test[:,i]), (N,N))/output_corr)
        output_err.append((np.reshape(test_outputs[:,i]-np.matmul(Ug, y_pred_test[:,i]), (N,N)))/output_corr)
        mre.append(np.linalg.norm(test_outputs[:,i] - np.matmul(Ug, y_pred_test[:,i]))/np.linalg.norm(test_outputs[:,i]))
        print("MRE: ", mre[-1])

    v_min = min(np.min(output_truth),np.min(output_pred))
    v_max = max(np.max(output_truth),np.max(output_pred))
    err_max = max(abs(np.min(output_err)), abs(np.max(output_err)))
    for i in range(3):
            ims.append(axs[0][i].pcolormesh(X,Y,input_truth[i],shading="gouraud",cmap="RdBu"))
            ims.append(axs[1][i].pcolormesh(X,Y,output_truth[i],shading="gouraud",cmap="RdBu",vmin=-1.5,vmax=1.5))
            ims.append(axs[2][i].pcolormesh(X,Y,output_pred[i],shading="gouraud",cmap="RdBu",vmin=-1.5,vmax=1.5))
            ims.append(axs[3][i].pcolormesh(X,Y,output_err[i],shading="gouraud",cmap="RdBu", vmin=-err_max,vmax=err_max))
            axs[3][i].set_title(f"MRE: {mre[i]:.3f}")
            for j in range(4):
                axs[j][i].set_aspect("equal","box")
                #axs[j][i].set_axis_off()
            i+=1
        
    #print viscosities as exponential
    axs[0][0].set_title(f"Viscosity: 10e{np.log10(vs[0]).round(3)}")
    axs[0][1].set_title(f"Viscosity: 10e{np.log10(vs[1]).round(3)}")
    axs[0][2].set_title(f"Viscosity: 10e{np.log10(vs[2]).round(3)}")
    subfigs[0].suptitle("Forcing Function",fontsize=16,y=0.95)
    subfigs[2].suptitle("Prediction",fontsize=16,y=0.95)
    subfigs[1].suptitle("Ground Truth",fontsize=16,y=0.95)
    subfigs[3].suptitle("Error",fontsize=16,y=0.95)

    plt.subplots_adjust(left=0.02,right=0.87,bottom=0.02,top=0.90)
    cax = []
    for i in range(4):
        temp = axs[i][-1].get_position()
        cax.append(subfigs[i].add_axes([0.89,temp.y0,0.02,temp.y1-temp.y0]))
        if i==0:
            cb = plt.colorbar(ims[i],cax=cax[i],ticks=[np.min(test_inputs),0,np.max(test_inputs)])
        elif i == 1:
            cb = plt.colorbar(ims[i],cax=cax[i],ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
        elif i == 2:
            cb = plt.colorbar(ims[i],cax=cax[i],ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
        elif i == 3:
            cb = plt.colorbar(ims[i],cax=cax[i],ticks=[-err_max,-0.4,-0.2,0,0.2,0.4,err_max])
        #cb.outline.set_visible(False)
        cb.ax.yaxis.set_tick_params(width=0.3)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(12)


    fig.savefig("NS-pca-vis-"+coloroption+".pdf")
else:
        
    y_pred_test = []
    for i in range(len(models)):
        models[i].to(device)
        y_pred_test.append(y_normalizer.decode(models[i](x_test).detach()).cpu().numpy().T)
        

    for model in range(len(models)):
        rel_err_nn_test = np.zeros(ntest)
        output_corr = np.zeros(ntest)
        for i in range(ntest):
            rel_err_nn_test[i] = np.linalg.norm(test_outputs[:, i]  - np.matmul(Ug, y_pred_test[model][:, i]))/np.linalg.norm(test_outputs[:, i])
        #plt.scatter(np.log10(v_test), rel_err_nn_test, 1)
        plt.xlabel("Viscosity")
        plt.xscale("log")
        plt.ylabel("Relative Error")
        rel_err_nn_test_fixed = []
        v_test_fixed = []
        for i in range(ntest):
            if rel_err_nn_test[i] < 1 and v_test[i] < 1:
                rel_err_nn_test_fixed.append(rel_err_nn_test[i])
                v_test_fixed.append(v_test[i])
                
        plt.scatter(v_test_fixed, rel_err_nn_test_fixed, 1)
        print("NN: ", str(model), "rel test error fixed", np.mean(rel_err_nn_test_fixed))
        plt.savefig("NS-pca-vis-mre"+str(model)+".pdf")
        
        plt.close("all")
        
        plt.xlabel("Viscosity")
        plt.xscale("log")
        plt.ylabel("Relative Error")
        plt.scatter(v_test, rel_err_nn_test, 1)
        print("NN: ", str(model), "rel test error full", np.mean(rel_err_nn_test))
        plt.savefig("NS-pca-vis-mre"+str(model)+"_full.pdf")
        
        plt.close("all")
    # plt.scatter(v[ntrain:ntrain+100], rel_err_nn_test[:100], 1)
    # z = np.polyfit(v[ntrain:], rel_err_nn_test, 2)
    # p = np.poly1d(z)
    # plt.plot(np.arange(0, 1, 0.01),p(np.arange(0, 1, 0.01)),"--")
    #plt.scatter(v[ntrain:], output_corr, 1)
    # plt.colorbar()
    
