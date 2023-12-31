import sys
import numpy as np
from mynn import *
from mydata import UnitGaussianNormalizer
from Adam import Adam
from timeit import default_timer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


M = int(sys.argv[1])
N_neurons = int(sys.argv[2])
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


acc = 0.999

xgrid = np.linspace(0,2*np.pi*(1 - 1.0/N), N)
xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]

inputs  = cs
outputs = K

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = np.reshape(inputs[:,:,:ntrain], (-1, ntrain))
    test_inputs  = np.reshape(inputs[:,:,ntrain:M], (-1, ntest))
    v_train = v[:ntrain]
    v_test  = v[ntrain:M]
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    # r_f = min(r_f, 512)
    # print(Si[98:210])
    r_f = 128
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
else:
    
    train_inputs =  theta[:ntrain, :]
    test_inputs  = theta[ntrain:M, :]
    r_f = N_theta
    x_train = torch.from_numpy(train_inputs.astype(np.float32))
    


train_outputs = np.reshape(outputs[:,:,:ntrain], (-1, ntrain))
test_outputs  = np.reshape(outputs[:,:,ntrain:M], (-1, ntest))
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)
r_g = np.argwhere(en_g<(1-acc))[0,0]
Ug = Uo[:,:r_g]
g_hat = np.matmul(Ug.T,train_outputs)
y_train = torch.from_numpy(g_hat.T.astype(np.float32))


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



layers = 4
for seed in range(10):
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    model = FNN(r_f+1, r_g, layers, N_neurons) 
    print(count_params(model))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = torch.nn.MSELoss(reduction='sum')
    y_normalizer.cuda()
    t0 = default_timer()
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

        torch.save(model, "PCANet_"+str(N_neurons)+"Nd_"+str(seed)+".model")
        scheduler.step()

        train_l2/= ntrain

        t2 = default_timer()
        print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)
    print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)


