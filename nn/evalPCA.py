import sys
import numpy as np
from mynn import *
from mydata import *
from datetime import datetime

import matplotlib as mpl 
from matplotlib.lines import Line2D 
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
plt.rc("text", usetex=True)         # Crisp axis ticks
plt.rc("font", family="serif")      # Crisp axis labels
plt.rc("legend", edgecolor='none')  # No boxes around legends

plt.rc("figure",facecolor="#ffffff")
plt.rc("axes",facecolor="#ffffff",edgecolor="#000000",labelcolor="#000000")
plt.rc("savefig",facecolor="#ffffff")
plt.rc("text",color="#000000")
plt.rc("xtick",color="#000000")
plt.rc("ytick",color="#000000")

color1 = 'tab:blue'
color2 = 'tab:green'
color3 = 'tab:orange'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def colnorm(u):
	return np.sqrt(np.sum(u**2,0))

# N = 256
# K = 200
# M = 2048

# xgrid = np.linspace(0,1,N)
# xgrid = xgrid[:-1]
# dx    = xgrid[1] - xgrid[0]

# # burgers param and data
# nu      = 0.01
# data    = np.load('../../data/N'+str(N)+'_K'+str(K)+'_M'+str(M)+'.npz')
# inputs  = data["inputs"]
# outputs = data["outputs"]


M = int(sys.argv[1])
#N_neurons = int(sys.argv[2])
#v_inv = int(sys.argv[3])
N = 64  # element
ntrain = int(M*.8)
ntest = M - ntrain
N_theta = 100
prefix = "../datagen/"
v = np.load(prefix+"viscosity_" + str(M) + ".npy")
theta = np.load(prefix+"theta_" + str(M) + ".npy")
cs = np.load(prefix+"curl_f_"  + str(M) + ".npy") 
K = np.load(prefix+"omega_"  + str(M) + ".npy")

print("Loaded data")
acc = 0.9999

xgrid = np.linspace(0,2*np.pi*(1 - 1.0/N), N)
dx    = xgrid[1] - xgrid[0]

inputs  = cs
outputs = K
print(inputs.shape)

compute_pca = False

train_inputs = np.reshape(inputs[:,:,:ntrain], (-1, ntrain))
test_inputs  = np.reshape(inputs[:,:,ntrain:M], (-1, ntest))
v_train = v[:ntrain]
v_test  = v[ntrain:M]

train_outputs = np.reshape(outputs[:,:,:ntrain], (-1, ntrain))
test_outputs  = np.reshape(outputs[:,:,ntrain:M], (-1, ntest))

if compute_pca:
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    # r_f = min(r_f, 512)
    # print(Si[98:210])
    #r_f = 128
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    np.save("PCANet_"+str(M)+"_Uf.npy", Uf)

    Uo,So,Vo = np.linalg.svd(train_outputs)
    en_g = 1 - np.cumsum(So)/np.sum(So)
    r_g = np.argwhere(en_g<(1-acc))[0,0]
    Ug = Uo[:,:r_g]
    g_hat = np.matmul(Ug.T,train_outputs)
    y_train = torch.from_numpy(g_hat.T.astype(np.float32))
    np.save("PCANet_"+str(M)+"_Ug.npy", Ug)
    
    
else:
    Uf = np.load("PCANet_"+str(M)+"_Uf.npy")
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    r_f = Uf.shape[1]

    Ug = np.load("PCANet_"+str(M)+"_Ug.npy")
    g_hat = np.matmul(Ug.T,train_outputs)
    y_train = torch.from_numpy(g_hat.T.astype(np.float32))
    r_g = Ug.shape[1]

print("PCA computed")

models = []
modelinfo = []
for learning_rate in [0.0001, 0.0003, 0.001, 0.003, 0.01]:
    for gamma in [0.05, 0.1, 0.3, 0.5]:
        for step_size in [50, 100, 150, 200, 250, 300, 350, 400, 500, 1000]:
            try:
                models.append(torch.load("../nn/PCANet_hyp_" + str(learning_rate) + "_" + str(gamma) + "_" + str(step_size) + ".model"))
                modelinfo.append((learning_rate, gamma, step_size))
            except:
                pass
print("Models loaded")

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_train = torch.from_numpy(np.hstack((x_train, np.reshape(v_train, (ntrain, -1))))).float()
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

if torch.cuda.is_available():
    x_normalizer.cuda()
    y_normalizer.cuda()

x_train = x_train.to(device)

# rel_err_nn_train = np.sum((y_pred_train-g_hat)**2,0)/np.sum(g_hat**2,0)
# mre_nn_train = np.mean(rel_err_nn_train)

# print(f_hat_test.shape)
# f_hat_test = np.matmul(Uf.T,test_inputs)
x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))
x_test = torch.from_numpy(np.hstack((x_test, np.reshape(v_test, (ntest, -1))))).float()
x_test = x_test.to(device)
y_pred_train = []
y_pred_test = []
for i in range(len(models)):
    models[i].to(device)
    y_pred_train.append(y_normalizer.decode(models[i](x_train).detach()).cpu().numpy().T)
    y_pred_test.append(y_normalizer.decode(models[i](x_test).detach()).cpu().numpy().T)
        
for model in range(len(models)):
    rel_err_nn_train = np.zeros(ntrain)
    for i in range(ntrain):
        rel_err_nn_train[i] = np.linalg.norm(train_outputs[:, i]  - np.matmul(Ug, y_pred_train[model][:, i]))/np.linalg.norm(train_outputs[:, i])
    mre_nn_train = np.mean(rel_err_nn_train)
    rel_err_nn_test = np.zeros(ntest)
    for i in range(ntest):
            rel_err_nn_test[i] = np.linalg.norm(test_outputs[:, i]  - np.matmul(Ug, y_pred_test[model][:, i]))/np.linalg.norm(test_outputs[:, i])
    mre_nn_test = np.mean(rel_err_nn_test)
    plt.scatter(v[ntrain:], rel_err_nn_test, 1)
    plt.xlabel("Viscosity")
    plt.ylabel("Mean Relative Error")
    print("NN: ", modelinfo[model], "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)

plt.savefig("NS-pca-vis-mre.pdf")
#plt.close("all")
# rel_err_nn_test = np.sum((y_pred_test-g_hat_test)**2,0)/np.sum(g_hat_test**2,0)
# mre_nn_test = np.mean(rel_err_nn_test)




# #########################################
# # save smallest, medium, largest
# test_input_save  = np.zeros((N,  N, 3))
# test_output_save = np.zeros((N,  N, 6))
# for i, ind in enumerate([np.argmin(rel_err_nn_test), np.argsort(rel_err_nn_test)[len(rel_err_nn_test)//2], np.argmax(rel_err_nn_test)]):
#     test_input_save[:, :, i]  = inputs[:, :, ntrain + ind]
#     # truth
#     test_output_save[:, :, i] = outputs[:, :, ntrain + ind]
#     # predict
#     test_output_save[:, :, i + 3] =  np.reshape(np.matmul(Ug, y_pred_test[:, ind]), (N,N))

# np.save(str(ntrain) + "_" + str(N_neurons) + "_test_input_save.npy",  test_input_save)
# np.save(str(ntrain) + "_" + str(N_neurons) + "_test_output_save.npy", test_output_save)





# loss_scale = 1000
# n_epochs = 500000
# for epoch in range(n_epochs):
# 	y_pred = model(x_train)
# 	loss = loss_fn(y_pred,y_train)*loss_scale

# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()
# 	if epoch % 1000 == 0:
# 		print("[{}/{}], loss: {}, time {}".format(epoch, n_epochs, np.round(loss.item(), 3),datetime.now()))
# 		torch.save(model, "PCANet_"+str(N_neurons)+".model")

	
# # # save the model
# # torch.save(model, "PCANet_"+str(N_neurons)+".model")

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.semilogy(rel_err_nn_train,lw=0.5,color=color1,label='training')
# ax.semilogy(rel_err_nn_test,lw=0.5,color=color2,label='test')
# ax.legend()
# plt.xlabel('data index')
# plt.ylabel('Relative errors')
# plt.tight_layout()
# plt.savefig('NN%d_errors.png' %(N_neurons),pad_inches=3)
# plt.close()

# ind = np.argmax(rel_err_nn_test)
# Y, X = np.meshgrid(xgrid, xgrid)

# fig,ax = plt.subplots(ncols=3, figsize=(9,3))
# vmin, vmax = min(test_outputs[:,ind]), max(test_outputs[:,ind])
# ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:,ind],(N,N)),               shading='gouraud')
# ax[1].pcolormesh(X, Y, np.reshape(test_outputs[:,ind],(N,N)),              shading='gouraud', vmin=vmin, vmax =vmax)
# ax[2].pcolormesh(X, Y, np.reshape(np.matmul(Ug,y_pred_test[:,ind]),(N,N)), shading='gouraud', vmin=vmin, vmax =vmax)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
# plt.savefig('worst_case_test_NN%d.png' %(N_neurons),pad_inches=3)
# plt.close()

# # for ind in np.random.randint(0,1024,(5,)):
# # 	fig,ax = plt.subplots(figsize=(3,3))
# # 	fig.subplots_adjust(bottom=0.2,left = 0.15)
# # 	ax.plot(xgrid,test_inputs[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
# # 	ax.plot(xgrid,test_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
# # 	ax.plot(xgrid,np.matmul(Ug,y_pred_test[:,ind]),lw=0.5,color=color3,label="NN u(T)")
# # 	ax.legend()
# # 	plt.xlabel('$x$')
# # 	plt.ylabel('u(x)')
# # 	plt.savefig('test'+str(ind)+'.png',pad_inches=3)
# # 	plt.close()

# ind = np.argmax(rel_err_nn_train)

# fig,ax = plt.subplots(ncols=3, figsize=(9,3))
# vmin, vmax = min(test_outputs[:,ind]), max(test_outputs[:,ind])
# ax[0].pcolormesh(X, Y, np.reshape(train_inputs[:,ind],(N,N)),               shading="gouraud")
# ax[1].pcolormesh(X, Y, np.reshape(train_outputs[:,ind],(N,N)),              shading="gouraud", vmin=vmin, vmax =vmax)
# ax[2].pcolormesh(X, Y, np.reshape(np.matmul(Ug,y_pred_train[:,ind]),(N,N)), shading="gouraud", vmin=vmin, vmax =vmax)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
# plt.savefig('worst_case_train_NN%d.png' %(N_neurons),pad_inches=3)
# plt.close()


# # for ind in np.random.randint(0,1024,(9,)):
# # 	fig,ax = plt.subplots(figsize=(3,3))
# # 	fig.subplots_adjust(bottom=0.2,left = 0.15)
# # 	ax.plot(xgrid,train_inputs[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
# # 	ax.plot(xgrid,train_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
# # 	ax.plot(xgrid,np.matmul(Ug,y_pred_train[:,ind]),lw=0.5,color=color3,label="NN u(T)")
# # 	ax.legend()
# # 	plt.xlabel('$x$')
# # 	plt.ylabel('u(x)')
# # 	plt.savefig('train'+str(ind)+'.png',pad_inches=3)
# # 	plt.close()


# # fig,ax = plt.subplots(figsize=(3,3))
# # fig.subplots_adjust(bottom=0.2,left = 0.15)
# # ax.semilogy(en_f,lw=0.5,color=color1,label='input')
# # ax.semilogy(en_g,lw=0.5,color=color2,label='output')
# # ax.legend()
# # plt.xlabel('index')
# # plt.ylabel('energy lost')
# # plt.savefig('training_PCA.png',pad_inches=3)
# # plt.close()
