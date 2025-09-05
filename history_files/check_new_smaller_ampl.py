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


def Stress_visco(tau, strain_rate_history, d2wd2e,sed):
    sum = d2wd2e + sed
    subIntegral = sum * strain_rate_history
    stress = np.trapz(subIntegral, tau)
    # print(f'stress = {stress}')
    return stress

def stress_MR_int(t, eps):
    t = t.ravel()
    # eps = DataWrapper.strain
    # E1 = DataWrapper.E
    # eta = DataWrapper.etha

    vect_len = t.size
    eps_rate = np.gradient(eps, t)
    dt = t[1] - t[0]
    t1 = t[1]


    sigma_vis_el = np.zeros(vect_len)

    d = eps ** 2 * (eps * (3.3460028787189144e-6 * eps - 0.0009997788898407148) + 0.16457425809043072)
    d2 = np.gradient(d, eps)
    d2Wd2e = np.gradient(d2, eps)

    # Видаляємо останні 2 точки
    d2Wd2e_new = d2Wd2e[:-2]

    # Вибираємо дві попередні точки
    x1, x2 = d2Wd2e_new[-2], d2Wd2e_new[-1]

    # Визначаємо індекси цих точок
    idx1, idx2 = len(d2Wd2e_new) - 2, len(d2Wd2e_new) - 1

    # Визначаємо уявну пряму через ці точки
    slope = (x2 - x1) / (idx2 - idx1)
    intercept = x1 - slope * idx1

    # Обчислюємо значення для двох видалених точок на основі прямої
    new_x1 = slope * len(d2Wd2e_new) + intercept
    new_x2 = slope * (len(d2Wd2e_new) + 1) + intercept

    # Додаємо ці значення до масиву
    d2Wd2e_new = np.append(d2Wd2e_new, [new_x1, new_x2])
    d2Wd2e = d2Wd2e_new

    # print(f'eps = {eps}')
    # print(f'd2 = {d2}')
    # print(f'd2wd2e = {d2Wd2e}')

    # Stress_visco(t[t<t1], sed(t[t<t1]))
    for i in range(vect_len):
        Tt = t1 - t
        # SED = sed(Tt.reshape(-1, 1))
        # ex = 0.1*np.exp(-0.1*Tt/eta)
        # SED = (-637.8875265273954*Tt - 637.8875265273954*np.exp(-Tt-np.exp(-85.4672872872024*Tt-1.8168101809208005)+ 0.5467780601705482))*0
        # SED = np.exp(13.99317635356684 - 1761.356446696241*Tt) *0
        # SED2 = -15.190272888077851*np.exp(-Tt) + np.exp(-56.404881532724873*Tt + 4.01837824525225)
        # SED = -0.0010997865234680451 * np.exp(8 * Tt ** 4 * (1.4454 - Tt))
        # SED = np.exp(30.176992064591458*Tt**2  - 68.342906402217124*Tt + 0.11737156346792166)
        SED = -0.0010997865234680451 * np.exp(8 * Tt ** 4 * (1.4454 - Tt))
        sigma_vis_el[i] = Stress_visco(t[t < t1], eps_rate[:t[t < t1].size], d2Wd2e[i], SED[:t[t < t1].size])
        t1 = t1 + dt
    return sigma_vis_el

y_n = stress_MR_int(X, eps)


fig, ax = plt.subplots()
ax.plot(eps, y_n, 'r', label='prediction')
# ax.plot(eps3, y_n, 'g', label='prediction')
# ax.scatter(lam_test[:, 0], y_test, c='g')
ax.scatter(eps, y, label='dataset')

ax.legend()
plt.xlabel('time')
plt.ylabel('Stress')
ax.grid('on')
plt.show()


fig, ax = plt.subplots()
# ax.plot(X, eps, 'r', label='prediction')
# ax.plot(eps3, y_n, 'g', label='prediction')
# ax.scatter(lam_test[:, 0], y_test, c='g')
ax.scatter(X, eps, label='dataset')

ax.legend()
plt.xlabel('time')
plt.ylabel('strain')
ax.grid('on')
plt.show()

