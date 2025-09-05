import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import *
from dso.program import Program
from dso import DeepSymbolicRegressor
from scipy.interpolate import CubicSpline


# file = r'D:\projects\data\loop_for_study_7.5hz6a.xls'
file = r'D:\projects\data\7.5h42alg.xls'
# file = r'D:\projects\data\7.5hz2alg.xls'
dataset = pd.read_excel(file)
dataset = dataset.iloc[:13, :3]

X_d = dataset['strain'].to_numpy()
y_d = dataset['stress'].to_numpy()
t_d = dataset['time'].to_numpy()

cs = CubicSpline(t_d, X_d)
cs_y = CubicSpline(t_d, y_d)
t = np.linspace(0, t_d[t_d.size-1], 50)
X = cs(t)
y = cs_y(t)

X = X.reshape(-1, 1)
E1 = 0.05
E2 = 0.1
etha = 0.02
print(t)
plt.scatter(X, y)
plt.grid('on')
plt.show()

Program.DataWrapper = DataWrapper(t, E1, etha)
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
    logger.info(f"Seed: {i}")
    logger.info(f"Model Pretty Output:\n{pretty_output}")

    fig, ax = plt.subplots()

    y_2 = model.predict(X)
    nrmse = np.sqrt(np.mean((y - y_2) ** 2))

    print(nrmse)
    logger.info(f"NRMSE: {nrmse}")

    ax.plot(X, y_2, 'r', label='prediction')
    ax.scatter(X_d, y_d, label='dataset')
    ax.legend()
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    ax.grid('on')
    plt.savefig(f'testimg{i}.png')
