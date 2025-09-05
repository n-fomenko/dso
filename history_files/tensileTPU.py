# import pandas as pd
# import matplotlib.pyplot as plt
#
# file_path = "D:/exper_data/TensileTPU_6.xls"
#
# data = pd.read_excel(file_path, sheet_name='Specimen 9')
#
# print(data.head())
#
# x = data.iloc[16120:19500, 0]
# y = data.iloc[16120:19500, 1]
# print(x)
# # Побудова графіка
# plt.figure(figsize=(8, 6))
# plt.plot(x, y)
# plt.xlabel('Nominal strain (%)')
# plt.ylabel('Standard force (MPa)')
# plt.grid(True)
# plt.show()

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import *
from dso.program import Program
from dso import DeepSymbolicRegressor
from scipy.interpolate import CubicSpline


file_path = "D:/exper_data/TensileTPU_6.xls"

data = pd.read_excel(file_path, sheet_name='Specimen 9', header=1)

# print(data.columns)

# data = data.iloc[19504:21194, :3]
data = data.iloc[19504:22884, :3]
data = data.iloc[::10]

data['Test time'] = pd.to_numeric(data['Test time'], errors='coerce')
data['Nominal strain'] = pd.to_numeric(data['Nominal strain'], errors='coerce')
data['Standard force'] = pd.to_numeric(data['Standard force'], errors='coerce')

data['Test time'] = data['Test time'].round(4)
X = data['Nominal strain'].to_numpy()
y = data['Standard force'].to_numpy()
t = data['Test time'].to_numpy()


X = X.reshape(-1, 1)
# print(y)
E1 = 0.05
# E2 = 0.1
etha = 0.02

plt.scatter(X, y)
plt.grid('on')
plt.show()

# Program.DataWrapper = DataWrapper(t, E1, etha)
Program.DataWrapper = DataWrapper("time", t, None, None)
Program.DataWrapper.FitStrainEnergy = True
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
    ax.scatter(X, y, label='dataset')
    ax.legend()
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    ax.grid('on')
    plt.savefig(f'testimg{i}.png')

