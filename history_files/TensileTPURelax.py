import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import DataWrapper
from dso.program import Program
from dso import DeepSymbolicRegressor
import datetime
from dso.cont_m import *

file_path = "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation5.TRA"
data =  np.genfromtxt(file_path, delimiter=';', dtype=None, encoding='utf-8', skip_header=8)


# indx = np.where(data[:,2])
X = data[:,0]
eps = data[:,2]/100
y = data[:,3]

# X = X[:,np.newaxis]


# New x values with 20 points
eps_new = np.ones(100) * eps.max()
eps_new[0] = 0
x_new = np.linspace(X.min(), X.max(), 100)

# Interpolating y values for the new x points
y_new = np.interp(x_new.flatten(), X.flatten(), y)

eps = eps_new
X = x_new
# X = X[:,np.newaxis]
eps = eps[:,np.newaxis]
y = y_new


# plt.scatter(X, y) # Use the reordered 'eps' and 'y'
# plt.grid('on')
# plt.show()

#
# plt.scatter(eps_new, y_new)
# plt.grid('on')
# plt.show()


E1 = 0.05
# E2 = 0.1
etha = 0.02
Is, trueStress, stretch, trueStrain = getIsSigaStretchTrueStrain(eps.flatten(), y) # for tau=tau(Is)

Program.DataWrapper = DataWrapper("time", X, None, None, None)
Program.DataWrapper.FitViscous = True
Program.DataWrapper.FitStrainEnergy = False
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
    model.fit(Is, y)
    print(model.program_.pretty())
    # Get the "pretty" output of the model
    pretty_output = model.program_.pretty()

    # Log the seed and pretty output
    logger.info(f"Seed: {i}")
    logger.info(f"Model Pretty Output:\n{pretty_output}")

    fig, ax = plt.subplots()

    y_2 = model.predict(X)
    nrmse = np.sqrt(np.mean((y - y_2) ** 2))

    print(nrmse)
    logger.info(f"NRMSE: {nrmse}")
    engStrain, engStress = ConvertTrueToEng(trueStrain, y_2) # for tau=tau(Is)
    ax.plot(engStrain, engStress, 'r', label='prediction')
    ax.scatter(X/100, y, label='dataset')
    # ax.plot(X, y_2, 'r', label='prediction')
    # ax.scatter(X, y, label='dataset')
    ax.legend()
    plt.xlabel('Strain')
    plt.ylabel('t')
    ax.grid('on')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'testimg{i}_{timestamp}.png')
    # plt.savefig(f'testimg2.png')
