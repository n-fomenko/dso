import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import *
from dso.program import Program
from dso import DeepSymbolicRegressor
import datetime

file_path = "D:/exper_data/TensileTPU_6.xls"
data = pd.read_excel(file_path, sheet_name='Specimen 9', header=1)

# print(data.columns)

data = data.iloc[19504:22884, :3]
print(data.size)
data = data.iloc[::24]

# print(data.dtypes)
data['Test time'] = pd.to_numeric(data['Test time'], errors='coerce')
data['Nominal strain'] = pd.to_numeric(data['Nominal strain'], errors='coerce')
data['Standard force'] = pd.to_numeric(data['Standard force'], errors='coerce')

data['Test time'] = data['Test time'].round(4)

X = np.column_stack((data['Test time'].to_numpy(), data['Nominal strain'].to_numpy()))

# Verify the shape and content of X
print(X.shape)
print(X[:5])

# eps = data['Nominal strain'].to_numpy()
y = data['Standard force'].to_numpy()
# X = data['Test time'].to_numpy()
print(data.dtypes)
print(data.size)
print('----------------')
# print(data.iloc[:-1, :])
# y += 0.2
# print(X[0])

X[:, 0] += -1255
X[:, 0] = X[:, 0]/100

print(X[:5])
# y += -5.0
# eps += -86.0
# print(data)

# X = X.reshape(-1, 1)

print(f"Length of t: {len(X[:, 0])}")
print(f"Length of eps: {len(X[:, 1])}")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
E1 = 0.05
# E2 = 0.1
etha = 0.02

print(f'X.type={type(X)}, y.type={type(y)}, eps.type={type(X[:, 1])}')
# print(f'X={X}, y={y}, eps={eps}')

plt.scatter(X[:, 1], y)
plt.grid('on')
plt.show()

Program.DataWrapper = DataWrapper(X[:, 1], E1, etha)
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

    ax.plot(X[:, 1], y_2, 'r', label='prediction')
    ax.scatter(X[:, 1], y, label='dataset')
    ax.legend()
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    ax.grid('on')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'testimg{i}_{timestamp}.png')
    # plt.savefig(f'testimg2.png')
