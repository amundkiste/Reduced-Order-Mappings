import sys
import numpy as np
from mynn import *
from mydata import UnitGaussianNormalizer
from Adam import Adam
from timeit import default_timer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


M = 5000
#N_neurons = int(sys.argv[2])
#v_inv = int(sys.argv[3])
N = 100  # element
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


train_inputs = np.reshape(inputs, (-1, M))
v_train = v
train_outputs = np.reshape(outputs, (-1, M))
try:
    Uf = np.load("PCANet_"+str(M)+"_Uf.npy")
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    r_f = Uf.shape[1]

    Ug = np.load("PCANet_"+str(M)+"_Ug.npy")
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
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
    np.save("PCANet_"+str(M)+"_Uf.npy", Uf)

    Uo,So,Vo = np.linalg.svd(train_outputs)
    en_g = 1 - np.cumsum(So)/np.sum(So)
    r_g = np.argwhere(en_g<(1-acc))[0,0]
    Ug = Uo[:,:r_g]
    g_hat = np.matmul(Ug.T,train_outputs)
    y_train = torch.from_numpy(g_hat.T.astype(np.float32))
    np.save("PCANet_"+str(M)+"_Ug.npy", Ug)
    
    print("PCA computed")

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_train = torch.from_numpy(np.hstack((x_train, np.reshape(v_train, (M, -1))))).float()
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

print("Input #bases : ", r_f, " output #bases : ", r_g)
 
################################################################
# training and evaluation
################################################################
batch_size = 32
print(x_train.dtype, y_train.dtype)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5


neurons = 200
layers = 4
for i in range(5):
    torch.manual_seed(i)
    np.random.seed(i)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        
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
        
        scheduler.step()
        train_l2/= ntrain

        losses[ep] = train_l2
        t2 = default_timer()
        if(ep%10 == 0):
            print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)
    torch.save(losses, "PCANet_"+str(i)+"_loss.npy")
    torch.save(model, "PCANet_"+str(i)+".model")
    print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)
    


