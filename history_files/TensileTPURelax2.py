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

X = data[:,0]
eps = data[:,2]/100
y = data[:,3]

strain_max = eps.max()
t_rise = X[np.argmax(eps)]
t_total = X.max()
extra_strain = 0.001 * strain_max  # приріст на 0,1% після t_rise

# Побудова strain
strain_phys_time = X.copy()
strain_phys = np.piecewise(
    strain_phys_time,
    [strain_phys_time <= t_rise, strain_phys_time > t_rise],
    [
        lambda t: (strain_max / t_rise) * t,
        lambda t: strain_max + extra_strain * (t - t_rise) / (t_total - t_rise)
    ]
)

N = 100
time = np.linspace(X.min(), X.max(), N)
strain = np.interp(time, strain_phys_time, strain_phys)
stress_exp = np.interp(time, X, y)

eps = strain
X = time
# eps = eps[:,np.newaxis]
y = stress_exp
# print("eps range:", eps.min(), eps.max())
plt.plot(X, eps)
plt.show()
Is, trueStress, stretch, trueStrain = getIsSigaStretchTrueStrain(eps.flatten(), y) # for tau=tau(Is)
# print(f'Is {Is[:5]}')
# print(f'true stress {trueStress[:5]}')
Program.DataWrapper = DataWrapper("time", X, None, None, None, eps)
Program.DataWrapper.FitStrainEnergy = False
Program.DataWrapper.FitViscous = True

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
# Is = Is[:, 1]
# Is = Is[:,np.newaxis]
for i in range(0, N_rep):
    model.config["experiment"]["seed"] = i
    model.fit(Is, trueStress) #лише від першого інваріанту
    print(model.program_.pretty())
    # Get the "pretty" output of the model
    pretty_output = model.program_.pretty()

    # Log the seed and pretty output
    logger.info(f"Seed: {i}")
    logger.info(f"Model Pretty Output:\n{pretty_output}")

    fig, ax = plt.subplots()

    y_2 = model.predict(Is)
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
