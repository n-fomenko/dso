import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import *
from dso.program import Program
from dso import DeepSymbolicRegressor

vector_len = 200
t = np.linspace(0.001, 1, vector_len)
eps_00 = 1.25
eps_0 = 1.75
# epsilon = eps_00 + (eps_0 - eps_00)*t/t[vector_len-1]
# eps = eps_0 * np.sin(2*np.pi * t / t[vector_len-1])
omega = 2*np.pi  # Angular frequency (from sine wave)
eps = eps_0 * np.sin(omega * t)

E1 = 0.05
E2 = 0.1
etha = 0.02


# dW_de = (lam**2 - 1/lam)*(C10 + C01/lam)

epsilon_rate = np.gradient(eps, t)

# Relaxation time
tau = etha / E2

sigma_sin = E1 + E2 * (tau * omega)**2 / (1 + (tau * omega) ** 2)
sigma_cos = E2 * (tau * omega) / (1 + (tau * omega) ** 2)

# sigma0 = E1 * eps_00  # Constant part of stress
y2 = eps_0*(sigma_sin * np.sin(omega * t) + sigma_cos * np.cos(omega * t)) - eps_0*sigma_cos


dataset = np.column_stack((t, y2))

X = dataset[:, :1]
y = dataset[:, 1]

plt.scatter(eps, y)
plt.grid('on')
plt.show()
Program.DataWrapper = DataWrapper(eps, E1, etha)
model = DeepSymbolicRegressor('test_conf_2.json')
N_rep = 1

import  logging
# Configure logging
logger = logging.getLogger("model_experiment")
logger.setLevel(logging.INFO)

# Create a file handler with UTF-8 encoding
file_handler = logging.FileHandler("../model_experiment.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

for i in range(0, N_rep):
    model.config["experiment"]["seed"] = i
    model.fit(X, y)
    print(model.program_.pretty())
    # Get the "pretty" output of the model
    pretty_output = model.program_.pretty()

    # Log the seed and pretty output
    logging.info(f"Seed: {i}")
    logging.info(f"Model Pretty Output:\n{pretty_output}")

    fig, ax = plt.subplots()

    y_2 = model.predict(X)
    nrmse = np.sqrt(np.mean((y - y_2) ** 2))

    print(nrmse)
    logging.info(f"NRMSE: {nrmse}")

    # y_t = model.predict()
    # print(y_2)
    # print(y_2.shape)
    # x_plot = strain_to_lam(lam_test[:, 0])
    # x_plot2 = strain_to_lam(x_train[:, 0])
    # print(x_plot.shape)
    # print(lam.size)
    ax.plot(eps, y_2, 'r', label='prediction')
    # ax.scatter(lam_test[:, 0], y_test, c='g')
    ax.scatter(eps, y, label='dataset')
    # ax.grid()
    ax.legend()
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    ax.grid('on')

    # plt.show()
    plt.savefig(f'testimg{i}.png')
