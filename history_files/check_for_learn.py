import matplotlib.pyplot as plt
import numpy as np
from dso import DeepSymbolicRegressor
from dso.program import Program
from dso.DataWrapper import *

vector_len = 50
t = np.linspace(0.001, 1, vector_len)
eps_00 = 1.25
eps_0 = 1.75
eps = eps_0*np.sin(2*np.pi*t/t[vector_len-1])
C10 = 0.1
C01 = 1.0
E1 = 0.05
E2 = 0.1
etha = 0.02

lam = eps + 1
dW_de = E1*eps
epsilon_rate = np.gradient(eps, t)

omega = 2*np.pi
tau = etha/E2

sigma_sin = E1 + E2 * (tau*omega)**2 / (1+(tau*omega)**2)
sigma_cos = E2*(tau*omega) / (1+(tau*omega)**2)

sigma0=E1*eps_00
y2 = eps_0*(sigma_sin*np.sin(omega*t)+sigma_cos*np.cos(omega*t))

dataset = np.column_stack((t, y2))

X = dataset[:, :1]
y = dataset[:, 1]
print(type(X))

Program.DataWrapper = DataWrapper(eps,E1,etha)
model = DeepSymbolicRegressor('test_conf_2.json')
N_rep = 5

for i in range(0, N_rep):
    model.config["experiment"]["seed"] = i
    model.fit(X, y)
    print(model.program_.pretty())

    fig, ax = plt.subplots()

    y_2 = model.predict(X)
    nrmse = np.sqrt(np.mean(y - y_2)**2)

    print(nrmse)

    ax.plot(eps, y_2, 'r', label='prediction')
    ax.scatter(eps, y, label='dataset')
    ax.legend()
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    ax.grid('on')

    plt.show()