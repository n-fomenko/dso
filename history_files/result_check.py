import numpy as np
import pandas as pd
from dso.DataWrapper import *
from dso.program import Program
from matplotlib import pyplot as plt

# file = r'D:\projects\data\loop_for_study_7.5hz6a_next10loops.xls'
# file2 = r'D:\projects\data\loop_for_study_7.5hz6a.xls'
# file = r'D:\projects\data\7.5h42alg.xls'
# dataset = pd.read_excel(file)
# dataset = dataset.iloc[82:,:3]
# eps = dataset['strain'].to_numpy()
# y = dataset['stress'].to_numpy()
# X = dataset['time'].to_numpy()
#
# dataset2 = pd.read_excel(file2)
# dataset2 = dataset2.iloc[:,:3]
# eps2 = dataset2['strain'].to_numpy()
# y2 = dataset2['stress'].to_numpy()
# X2 = dataset2['time'].to_numpy()

file3 = r'D:\projects\data\7.5hz2alg.xls'
dataset3 = pd.read_excel(file3)
dataset3 = dataset3.iloc[1:13,:3]
eps3 = dataset3['strain'].to_numpy()
y3 = dataset3['stress'].to_numpy()
X3 = dataset3['time'].to_numpy()


# file4 = r'D:\projects\data\7.5hz6a.xls'
# dataset4 = pd.read_excel(file4)
# dataset4 = dataset4.iloc[56:,:3]
# eps4 = dataset4['strain'].to_numpy()
# y4 = dataset4['stress'].to_numpy()
# X4 = dataset4['time'].to_numpy()

# print(dataset[137:139])
X3 = X3.reshape(-1, 1)
E1 = 0.05
E2 = 0.1
etha = 0.02

# plt.scatter(eps, y)
# plt.scatter(eps2, y2, c='g')
plt.scatter(eps3, y3, c='m')
# plt.scatter(eps4, y4, c='r')
plt.grid('on')
plt.show()
# Program.DataWrapper = DataWrapper(eps, E1, etha)


def Stress_visco(tau, strain_rate_history, d2wd2e,sed):
    sum = d2wd2e + sed*0
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
    # d = (1/2)*E1*np.power(eps,2)
    # d = 60.24626324213107 * np.power(eps, 2) * (9.488588273963604 * np.power(eps, 2) + 8.941642543303079)
    # d = np.power(eps, 2) * (1963.2387118689746 * np.power(eps, 3) + 26.125137091749578)
    # d = 3.2198877035343236*np.power(eps, 2)*(135.3465303459846*np.power(eps, 2)+8.3154819754301949)

    # d = 35.24991028831693 * np.power(eps, 3) * (1.5977850872318107 * eps + 4.84219393296361)
    d = 897.8866433690374*np.power(eps, 2)*(eps-0.1334616490015489)*(eps + 0.12167165601485434)
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
        SED = np.exp(13.99317635356684 - 1761.356446696241*Tt) *0
        SED2 = -15.190272888077851*np.exp(-Tt) + np.exp(-56.404881532724873*Tt + 4.01837824525225)
        sigma_vis_el[i] = Stress_visco(t[t < t1], eps_rate[:t[t < t1].size], d2Wd2e[i], SED[:t[t < t1].size])
        t1 = t1 + dt
    return sigma_vis_el

y_n = stress_MR_int(X3, eps3)

# Create a figure with 2 subplots arranged in a column
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
#
# # First subplot
# axes[0].plot(X3, eps3, 'r', label='prediction')
# axes[0].scatter(X3, eps3, label='dataset')
#
# axes[0].legend()
# axes[0].set_xlabel('time')
# axes[0].set_ylabel('Strain')
# axes[0].grid('on')

# # Second subplot
# axes[1].plot(X[:135], y_n[:135], 'r', label='prediction')
# axes[1].plot(X[135:], y_n[135:], 'g', label='prediction')
# axes[1].scatter(X, y, label='dataset')
# axes[1].legend()
# axes[1].set_xlabel('time')
# axes[1].set_ylabel('Stress')
# axes[1].grid('on')

# Adjust layout
# plt.tight_layout()
#
# plt.show()
#
# fig, ax = plt.subplots()
# # ax.plot(eps[:135], y_n[:135], 'r', label='prediction')
# # ax.plot(eps[135:], y_n[135:], 'g', label='prediction')
# # # ax.scatter(lam_test[:, 0], y_test, c='g')
# # ax.scatter(eps, y, label='dataset')
#
# ax.legend()
# plt.xlabel('Strain')
# plt.ylabel('Stress')
# ax.grid('on')
# plt.show()

fig, ax = plt.subplots()
ax.plot(eps3, y_n, 'r', label='prediction')
# ax.plot(eps3, y_n, 'g', label='prediction')
# ax.scatter(lam_test[:, 0], y_test, c='g')
ax.scatter(eps3, y3, label='dataset')

ax.legend()
plt.xlabel('Strain')
plt.ylabel('Stress')
ax.grid('on')
plt.show()



