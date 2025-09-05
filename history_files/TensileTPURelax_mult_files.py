import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sp
from dso.DataWrapper import DataWrapper
from dso.program import Program
from dso import DeepSymbolicRegressor
import datetime
from dso.cont_m import *


def process_file_tpu(file_path, N=100):
    # Read raw data
    data = np.genfromtxt(file_path, delimiter=';', dtype=None, encoding='utf-8', skip_header=8)

    X = data[:, 0]
    eps = data[:, 2] / 100
    y = data[:, 3]

    # Find parameters for physical strain build
    strain_max = eps.max()
    t_rise = X[np.argmax(eps)]
    t_total = X.max()
    extra_strain = 0.001 * strain_max     # +0.1% after t_rise

    # Build strain over physical time
    strain_phys_time = X.copy()
    strain_phys = np.piecewise(
        strain_phys_time,
        [strain_phys_time <= t_rise, strain_phys_time > t_rise],
        [
            lambda t: (strain_max / t_rise) * t,
            lambda t: strain_max + extra_strain * (t - t_rise) / (t_total - t_rise)
        ]
    )

    # Interpolation
    time = np.linspace(X.min(), X.max(), N)
    strain = np.interp(time, strain_phys_time, strain_phys)
    stress_exp = np.interp(time, X, y)


    # Return as N×3 array
    return np.column_stack((time, strain, stress_exp))


# Paths to your two files
file_path1 = "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation5.TRA"
file_path2 = "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation10.TRA"

# Process both
data1 = process_file_tpu(file_path1, N=100)
data2 = process_file_tpu(file_path2, N=100)

# Combine into one dataset
dataset = np.vstack((data1, data2))

print(dataset.shape)    # (200, 3) if N=100 for each
print(dataset[:5])      # First few rows
print(dataset[-5:])     # Last few rows

X = dataset[:,:2]
# X = X[:,np.newaxis]
eps = dataset[:,1]
y = dataset[:,2]

# print(X[:,0])
plt.scatter(X[:,0], y)
plt.show()

Is, trueStress, stretch, trueStrain = getIsSigaStretchTrueStrain(eps, y) # for tau=tau(Is)

# print("Is shape:", Is.shape)  # should be (N, 3)
# print("Any NaN in Is?", np.isnan(Is).any())
# print("Any Inf in Is?", np.isinf(Is).any())
# for k in range(Is.shape[1]):
#     print(f"Is col {k}: min={Is[:,k].min()} max={Is[:,k].max()}")

Program.DataWrapper = DataWrapper("strain", eps, None, None, None)
Program.DataWrapper.FitViscous = True
Program.DataWrapper.FitStrainEnergy = False
model = DeepSymbolicRegressor('D:/projects/dso/test_conf_2.json')
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
from matplotlib import cm

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

    y_2 = model.predict(X[100:, :])
    y_3 = model.predict(X[:100, :])
    # nrmse = np.sqrt(np.mean((y - y_2) ** 2))
    #
    # print(nrmse)
    # logger.info(f"NRMSE: {nrmse}")
    engStrain, engStress = ConvertTrueToEng(trueStrain[100:], y_2) # for tau=tau(Is)
    engStrain3, engStress3 = ConvertTrueToEng(trueStrain[:100], y_3)  # for tau=tau(Is)
    # ax.plot(engStrain, engStress, 'r', label='prediction')
    # ax.scatter(X/100, y, label='dataset')
    ax.plot(X[:100,0], engStress3, 'r', label='prediction')
    ax.scatter(X[:100,0], y[:100], label='dataset')

    ax.plot(X[100:,0], engStress, 'r', label='prediction')
    ax.scatter(X[100:,0], y[100:], label='dataset')

    ax.legend()
    plt.xlabel('t')
    plt.ylabel('stress')
    ax.grid('on')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'testimg{i}_{timestamp}.png')
    plt.show()


    # # Plot the surface.
    # surf = ax.plot_surface(X[:,0], X[:,1], engStress, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # plt.show()
    # plt.savefig(f'testimg2.png')
