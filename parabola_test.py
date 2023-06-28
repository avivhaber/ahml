from ahml.nn import MLP, total_loss
import numpy as np
import matplotlib.pyplot as plt

# Attempts to learn the parabola y=x^2
N = 100
B = 1

net = MLP(1, 1, [6, 6], lr=0.01)

for epoch in range(100):
    Xs = np.linspace(-2, 2, N)
    np.random.shuffle(Xs)
    Ys = (Xs**2).tolist()
    Xs = Xs.tolist()
    for i in range(N // B):
        Xb = Xs[B*i : B*(i+1)]
        Yb = Ys[B*i : B*(i+1)]

        outs = [net([X]) for X in Xb]
        loss = total_loss(outs, [[y] for y in Yb])

        net.zero_grad()
        loss.backward()
        net.update()

        print(f"loss: {loss}")

Xtest = np.linspace(-2, 2, 100)
Ytest = Xtest**2
Xtest = Xtest.tolist()
Youts = [net([X])[0].val for X in Xtest]

fig, ax = plt.subplots()
ax.plot(Xtest, Ytest)
ax.plot(Xtest, Youts)
plt.show()