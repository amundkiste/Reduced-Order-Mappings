import sys
import numpy as np
from mynn import *
from mydata import UnitGaussianNormalizer
from Adam import Adam
from timeit import default_timer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


acc = 0.9999

xgrid = np.linspace(0,2*np.pi*(1 - 1.0/N), N)
xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]

inputs  = cs
outputs = K

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
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    r_f = Uf.shape[1]

    Ug = np.load("PCANet_"+str(M)+"_Ug.npy")
    g_hat = np.matmul(Ug.T,train_outputs)
    y_train = torch.from_numpy(g_hat.T.astype(np.float32))
    r_g = Ug.shape[1]

print("PCA computed")

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_train = torch.from_numpy(np.hstack((x_train, np.reshape(v_train, (ntrain, -1))))).float()
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

print("Input #bases : ", r_f, " output #bases : ", r_g)
 
################################################################
# training and evaluation
################################################################
batch_size = 16
print(x_train.dtype, y_train.dtype)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5


neurons = 200
layers = 4
for learning_rate in  [0.0001, 0.0003, 0.001, 0.003, 0.01]:
    for gamma in [0.01, 0.03, 0.1]:
        for step_size in [250, 300, 350, 400, 500, 1000]:
            try:
                model = torch.load("PCANet_hyp_"+str(learning_rate)+"_"+str(gamma)+"_"+str(step_size)+".model")
                print("Skipping: ", "PCANet_hyp_"+str(learning_rate)+"_"+str(gamma)+"_"+str(step_size)+".model")
                continue
            except:
                print("Training: ", "PCANet_hyp_"+str(learning_rate)+"_"+str(gamma)+"_"+str(step_size)+".model")
                pass
            torch.manual_seed(1)
            np.random.seed(1)
                
            model = FNN(r_f+1, r_g, layers, neurons) 
            print(count_params(model))
            model.to(device)

            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            myloss = torch.nn.MSELoss(reduction='sum')
            y_normalizer.cuda()
            t0 = default_timer()
            losses = np.zeros(epochs)
            for ep in range(epochs):
                model.train()
                t1 = default_timer()
                train_l2 = 0
                for x, y in train_loader:
                    x, y = x.cuda(), y.cuda()

                    batch_size_ = x.shape[0]
                    optimizer.zero_grad()
                    out = model(x)
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                    loss = myloss(out , y)
                    loss.backward()
                    optimizer.step()
                    train_l2 += loss.item()
                
                losses[ep] = train_l2
                scheduler.step()
                train_l2/= ntrain

                t2 = default_timer()
                if(ep%50 == 0):
                    print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)
            torch.save(losses, "PCANet_hyp_"+str(learning_rate)+"_"+str(gamma)+"_"+str(step_size)+".loss")
            torch.save(model, "PCANet_hyp_"+str(learning_rate)+"_"+str(gamma)+"_"+str(step_size)+".model")
            print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)
            


