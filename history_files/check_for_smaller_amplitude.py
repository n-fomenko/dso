import datetime
import pandas as pd
from matplotlib import pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import *
from dso.program import Program
from dso import DeepSymbolicRegressor
import numpy as np

file_path = "D:/exper_data/TensileTPU_7.xls"

data = pd.read_excel(file_path, sheet_name='Specimen 10', header=1)

print(data.head())
data['Test time'] = pd.to_numeric(data['Test time'], errors='coerce')
data['Nominal strain'] = pd.to_numeric(data['Nominal strain'], errors='coerce')
data['Standard force'] = pd.to_numeric(data['Standard force'], errors='coerce')

data['Test time'] = data['Test time'].round(4)

data_n= data[(data['Nominal strain']<80) & (data['Standard force'] < 5.7)]

print(data_n.size)
data_n = data_n.iloc[15000:17300, :]
data_n = data_n.iloc[::20]
eps = data_n['Nominal strain'].to_numpy()
y = data_n['Standard force'].to_numpy()
X = data_n['Test time'].to_numpy()
X = X.reshape(-1, 1)

plt.scatter(eps, y)
plt.grid('on')
plt.show()
E1 = 0.05
# E2 = 0.1
etha = 0.02


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
    logger.info("check for smaller amplitude")
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
