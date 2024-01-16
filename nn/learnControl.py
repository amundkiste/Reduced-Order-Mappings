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

train_inputs = np.reshape(inputs[:,:,:ntrain], (-1, ntrain))
print(train_inputs.shape)
v_train = v[:ntrain]
compute_input_PCA = True
x_train = torch.from_numpy(np.vstack((train_inputs, np.reshape(v_train, (-1, ntrain)))).transpose()).float()
print(x_train.shape)
train_outputs = np.reshape(outputs[:,:,:ntrain], (ntrain, -1))
y_train = torch.from_numpy(train_outputs).float()
print(y_train.shape)
 
################################################################
# training and evaluation
################################################################
batch_size = 16
print(x_train.dtype, y_train.dtype)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

learning_rate = 0.0001

epochs = 500
step_size = 100
gamma = 0.5


neurons=500
layers = 4
model = FNN(64*64+1, 64*64, layers, neurons) 
print(count_params(model))
model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = torch.nn.MSELoss(reduction='sum')
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
        loss = myloss(out , y)
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    torch.save(model, "PCANet_control.model")
    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    if(ep%50 == 0):
        print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)
print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)


