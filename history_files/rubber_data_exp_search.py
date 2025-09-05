import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import *
from dso.program import Program
from dso import DeepSymbolicRegressor
import datetime

from scipy.interpolate import CubicSpline

# file = r'D:\projects\data\loop_for_study_7.5hz6a.xls'
# file = r'D:\projects\data\7.5h42alg.xls'
file = r'D:\projects\data\7.5hz2alg.xls'
dataset = pd.read_excel(file)
dataset = dataset.iloc[1:, :3]
eps_d = dataset['strain'].to_numpy()
y_d = dataset['stress'].to_numpy()
t_d = dataset['time'].to_numpy()



cs = CubicSpline(t_d, eps_d)
cs_y = CubicSpline(t_d, y_d)
X = np.linspace(0, t_d[t_d.size-1], 50)
eps = cs(X)
y = cs_y(X)

X = X.reshape(-1, 1)

print(dataset.head())
print(f'X.type={type(X)}, y.type={type(y)}, eps.type={type(eps)}')
print(f'X={X}, y={y}, eps={eps}')

E1 = 0.05
E2 = 0.1
etha = 0.02
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
    logger.info(f"Seed: {i}")
    logger.info(f"Model Pretty Output:\n{pretty_output}")

    fig, ax = plt.subplots()

    y_2 = model.predict(X)
    nrmse = np.sqrt(np.mean((y - y_2) ** 2))

    print(nrmse)
    logger.info(f"NRMSE: {nrmse}")

    ax.plot(eps, y_2, 'r', label='prediction')
    ax.scatter(eps, y, label='dataset')
    ax.legend()
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    ax.grid('on')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'testimg{i}_{timestamp}.png')
    # plt.savefig(f'testimg2.png')
