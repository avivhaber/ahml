import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Attempts to learn the parabola y=x^2
N = 100
B = 1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

Xs = torch.linspace(0, 2, N)
Ys = Xs**2
data = torch.utils.data.TensorDataset(Xs, Ys)
loader = torch.utils.data.DataLoader(data, batch_size=B, shuffle=True)

for epoch in range(100):
    for Xb,Yb in loader:
        outs = net(Xb.reshape(B,1))
        loss = criterion(outs, Yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss: {loss}")

Xtest = torch.linspace(0, 2, 100)
Ytest = Xtest**2
Youts = [net(torch.tensor([x])) for x in Xtest]

fig, ax = plt.subplots()
ax.plot(Xtest, Ytest)
ax.plot(Xtest, [y.detach().numpy() for y in Youts])
plt.show()