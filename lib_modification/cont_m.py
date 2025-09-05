import csv
from symbol import return_stmt
from lib_modification.DataWrapper import DataWrapper
import numpy as np
import matplotlib.pyplot as plt
import os


# Read in csv file

def ReadData(path, ExpType):
    file = open(path)

    csvReader = csv.reader(file, delimiter=';')

    rows = []

    for row in csvReader:
        rows.append(row)

    rows = np.array(rows, dtype=np.float64)

    file.close()

    return rows


# Convert engineering stress to true stress

def ConvertEngToTrue(engStrain, engStress):
    trueStrain = np.log(engStrain + 1.)

    trueStress = engStress * (1. + engStrain)

    return [trueStrain, trueStress]


# Convert true stress to engineering stress

def ConvertTrueToEng(trueStrain, trueStress):
    engStrain = np.exp(trueStrain) - 1.

    engStress = trueStress / (1. + engStrain)

    return [engStrain, engStress]


# Calculate deformation gradient F from stretches l1, l2 and l3

def GetDeformationGradient(l1, l2, l3):
    F = np.zeros((len(l1), 3, 3))
    l1 = l1.flatten()
    l2 = l2.flatten()
    l3 = l3.flatten()
    F[:, 0, 0] = l1

    F[:, 1, 1] = l2

    F[:, 2, 2] = l3

    return F


# Determine invariants of deformation gradient F

def GetInvariants(F):
    C = GetRightCauchyGreen(F)

    I1 = np.trace(C, axis1=1, axis2=2)

    I2 = 0.5 * (np.trace(C, axis1=1, axis2=2) ** 2 - np.trace(np.linalg.matrix_power(C, 2), axis1=1, axis2=2))

    J = np.linalg.det(F)

    return [I1, I2, J]


# Calculate left Cauchy-Green tensor from F

def GetLeftCauchyGreen(F):
    return F @ np.swapaxes(F, -2, -1)


# Calculate right Cauchy-Green tensor from F

def GetRightCauchyGreen(F):
    return np.swapaxes(F, -2, -1) @ F


# Get invariants and sigma for input

def getIsSigaStretchTrueStrain(eps, s):
    # Provide strain eps in floats (not in percent) and 1st Piola Kirchhoff stress s in MPa
    eps = eps.flatten()
    trueStrain, trueStress = ConvertEngToTrue(eps, s)

    # Calculate deformation gradient

    stretch = eps + 1
    print(f'stretch shape = {stretch.shape}')

    F = GetDeformationGradient(stretch, 1. / np.sqrt(stretch), 1. / np.sqrt(stretch))

    # Get invariant

    I1, I2, J = GetInvariants(F)

    Is = np.stack([I1, I2, J]).transpose()

    return Is, trueStress, stretch, trueStrain


# Predict model response

def predictStress(dPsidI1, dPsidI2, stretch, Is):
    modelresponse = 2 * (dPsidI1 + dPsidI2 * Is[:, 0]) * (stretch ** 2 - 1 / stretch) - 2 * dPsidI2 * ( # 5.45 p.223 bergstrom
                stretch ** 4 - 1 / (stretch ** 2))

    return modelresponse

#
# # Read in treloar file
#
# TreloarUT = ReadData("C:/Users/Nataliia/Downloads/test.csv", 'UT')
#
# # Extract strain and stress
#
# eps, s = TreloarUT[:, 0] / 100, TreloarUT[:, 1]
#
# # Determine invariants
#
# Is, trueStress, stretch, trueStrain = getIsSigaStretchTrueStrain(eps, s)
#
# # Calculate fitted Mooney-Rivlin response for validation
#
# modelresponse = predictStress(2.0193, .3075, stretch, Is)
#
# # Convert to engineering stresses and strains
#
# engStrain, engStress = ConvertTrueToEng(trueStrain, modelresponse)
#
# # Plot data versus Mooney-Rivlin prediction
#
# plt.figure(1)
#
# plt.plot(engStrain, engStress, 'r*', label='UT Exp')
#
# plt.plot(eps, s, 'bo', label='UT Exp')
#
# plt.show()



def getIs(strain):
    stretch = strain + 1.
    F = GetDeformationGradient(stretch, 1. / np.sqrt(stretch), 1. / np.sqrt(stretch))
    I1, I2, J = GetInvariants(F)
    Is = np.stack([I1, I2, J]).T
    return Is, stretch


def W_expr(Is):
    I1 = Is[:, 0]
    I2 = Is[:, 1]
    J  = Is[:, 2]

    # return J + np.log(np.exp(0.8396667 + 0.27462308 * I2) - 1.27930107)
    # return I1 + np.log(I1 + J*np.log(I2) + 2*J + np.log(J))
    # return -I1 - I2*(3.5233090007497196 - np.exp(0.06625758859316393*J*(I2 + 23.608552517611187)))+1.0001485092944702
    # return I2*(-I1 + 0.2668726753980835*I2*(I2 - 3.9744773735536034) + I2 + 1.782835215408525)
    # return I2*(-4.420533369702091*I1 + 4.420533369702091*I2*np.log(0.9794310893305753*I2 - 1.4416378158483538) - 5.818700391760992)

    # return I2*(-4.420533369702091*I1 + 4.420533369702091*I2*np.log(0.9794310893305753*I2 - 1.4416378158483538) - 5.818700391760992)

    return  6.7262628848377842 * (
    0.4742919386246813 * np.log(J * np.log(11.516720835774006 - np.exp(I1)) + 2.5552182739329368)
    + 14.878723528173166
) + 1


def calculate_gradient_safe(W, I):
    dW = np.zeros_like(W)
    dI = np.diff(I)

    # Обробка першого елемента
    dW[0] = (W[1] - W[0]) / (I[1] - I[0]) if I[1] != I[0] else 0.0

    # Обробка проміжних елементів
    for i in range(1, len(W) - 1):
        dI_i = I[i+1] - I[i-1]
        if dI_i != 0:
            dW[i] = (W[i+1] - W[i-1]) / dI_i
        else:
            dW[i] = 0.0 # Градієнт дорівнює нулю, якщо значення не змінюється

    # Обробка останнього елемента
    dW[-1] = (W[-1] - W[-2]) / (I[-1] - I[-2]) if I[-1] != I[-2] else 0.0

    return dW

# def LVE_expr(time, strain, sed):
# def LVE_expr(Is, time, sed, DataWrapper=None):
#     N = len(time)
#
#     stress = np.zeros(N)
#     # tau_raw = sed(Is) # від I
#     # tau = np.exp(np.clip(tau_raw, -10, 10))
#
#     # Is1 = Is[:, 0]
#     # Is1 = Is1[:, np.newaxis]
#     # tau2 = sed(Is)
#     # tau2 = np.abs(tau)
#     # g = [0.27]
#     # tau = [300.0]
#     g = [0.1252, 0.19]
#     # g = [0.1252]
#     tau = [110.85, 1068]
#
#     # g = [0.0577, 0.173, 0.108]
#     # tau = [42.45, 15260, 221.9]
#     stressV = np.zeros(len(g))
#
#     # strain = DataWrapper.strain
#     # strain = strain.flatten() # для tau = sed(strain)
#     #
#     # Is, stretch = getIs(strain)
#     #print(f'stretch shape = {stretch.shape}')
#     W = W_expr(Is)
#     # print(W.shape)
#
#
#     # W_sym = (
#     #         -I1
#     #         - I2 * (
#     #                 3.5233090007497196
#     #                 - sp.exp(0.06625758859316393 * J * (I2 + 23.608552517611187))
#     #         )
#     #         + 1.0001485092944702
#     # )
#     #
#     # # (опційно) Спрощення
#     # W_sym = sp.simplify(W_sym)
#
#     # W_sym = I1 + sp.log(I1 + J * sp.log(I2) + 2 * J + sp.log(J))
#
#     # Побудова виразу
#     # W_sym = I2 * (
#     #         -I1 +
#     #         0.2668726753980835 * I2 * (I2 - 3.9744773735536034) +
#     #         I2 +
#     #         1.782835215408525
#     # )
#     # W_sym = I2*(-4.420533369702091*I1 + 4.420533369702091*I2*sp.log(0.9794310893305753*I2 - 1.4416378158483538) - 5.818700391760992)
#     # # Спрощення, якщо потрібно:
#     # # W = sp.simplify(W)
#     #
#     # # Виведення
#     # # print("W =", W)
#     #
#     # # Аналітичні похідні
#     # dW_dI1_sym = sp.diff(W_sym, I1)
#     # dW_dI2_sym = sp.diff(W_sym, I2)
#     #
#     # # Перевіримо вирази
#     # # print("dW/dI1:", dW_dI1_sym)
#     # # print("dW/dI2:", dW_dI2_sym)
#     #
#     # # Створення чисельних функцій
#     # f_dW_dI1 = sp.lambdify((I1, I2, J), dW_dI1_sym, 'numpy')
#     # f_dW_dI2 = sp.lambdify((I1, I2, J), dW_dI2_sym, 'numpy')
#     #
#     # I1_vals = Is[:, 0]
#     # I2_vals = Is[:, 1]
#     # J_vals = Is[:, 2]
#     #
#     # dW_dI1 = f_dW_dI1(I1_vals, I2_vals, J_vals)
#     # dW_dI2 = f_dW_dI2(I1_vals, I2_vals, J_vals)
#
#     I1 = Is[:, 0]
#     I2 = Is[:, 1]
#
#
#     dW_dI1 = np.gradient(W, I1, edge_order=2)
#     dW_dI2 = np.gradient(W, I2, edge_order=2)
#
#     # dW_dI1 = calculate_gradient_safe(W, I1)
#     # dW_dI2 = calculate_gradient_safe(W, I2)
#
#     stress_all = predictStress(dW_dI1, dW_dI2, stretch, Is)
#
#     engStrain, engStress = ConvertTrueToEng(strain, stress_all)
#
#     stressH0 = engStress[0]
#     # print(engStress)
#
#     # plt.scatter(strain, stress_all)
#     # plt.show()
#
#     for i in range(1, N):
#         stressH1 = engStress[i]
#         dstressH = stressH1 - stressH0
#         dt = time[i] - time[i - 1]
#
#         #print(f"Ітерація: {i}, dt: {dt}, dstressH: {dstressH}, stressH0: {stressH0}, stressH1: {stressH1}")
#
#         stress[i] = stressH1
#
#         for j in range(len(g)):
#             stressV[j] = (
#                     np.exp(-dt / tau[j]) * stressV[j] + # має бути 1 g, а тау ітерується по i
#                     g[j] * stressH0 * (1 - np.exp(-dt / tau[j])) +
#                     g[j] * dstressH / dt * (dt - tau[j] + tau[j] * np.exp(-dt / tau[j]))
#             )
#             stress[i] -= stressV[j]
#
#         stressH0 = stressH1
#
#
#     return stress

# def LVE_expr(time, strain, sed):
#     N = len(time)
#     stress = np.zeros(N)
#     # g = [0.19]
#
#     g = [0.1252, 0.19]
#     tau = [110.85, 1068]
#
#     # g = [0.0577, 0.173, 0.108]
#     # tau = [42.45, 15260, 221.9]
#     stressV = np.zeros(len(g))
#
#     Is, stretch = getIs(strain)
#     # tau = sed(Is)
#     # tau = 500 + tau
#
#     W = W_expr(Is)
#     I1 = Is[:, 0]
#     I2 = Is[:, 1]
#
#     dW_dI1 = np.gradient(W, I1, edge_order=2)
#     dW_dI2 = np.gradient(W, I2, edge_order=2)
#
#     stress_all = predictStress(dW_dI1, dW_dI2, stretch, Is)
#
#     engStrain, engStress = ConvertTrueToEng(strain, stress_all)
#
#     stressH0 = engStress[0]
#
#     for i in range(1, N):
#         stressH1 = engStress[i]
#         dstressH = stressH1 - stressH0
#         dt = time[i] - time[i - 1]
#         stress[i] = stressH1
#
#         for j in range(len(g)):
#             stressV[j] = (
#                     np.exp(-dt / tau[j]) * stressV[j] +
#                     g[j] * stressH0 * (1 - np.exp(-dt / tau[j])) +
#                     g[j] * dstressH / dt * (dt - tau[j] + tau[j] * np.exp(-dt / tau[j]))
#             )
#             stress[i] -= stressV[j]
#
#         stressH0 = stressH1
#
#     return stress

# def LVE_expr(time, strain, sed_list):
#     for sed in sed_list:
#         N = len(time)
#         # time = time[:,0]
#         # N = 100
#         stress = np.zeros(N)
#         # g = [0.1252, 0.19]
#         # tau = [110.85, 1068]
#         g = [0.19]
#         # tau = [500]
#         # g = [0.0577, 0.173, 0.108]
#         # tau = [42.45, 15260, 221.9]
#         stressV = np.zeros(len(g))
#
#         Is, stretch = getIs(strain)
#         tau = sed(Is)
#         tau = 500 + tau
#
#         W = W_expr(Is)
#         I1 = Is[:, 0]
#         I2 = Is[:, 1]
#
#         dW_dI1 = np.gradient(W, I1, edge_order=2)
#         dW_dI2 = np.gradient(W, I2, edge_order=2)
#
#         stress_all = predictStress(dW_dI1, dW_dI2, stretch, Is)
#
#         engStrain, engStress = ConvertTrueToEng(strain, stress_all)
#
#         stressH0 = engStress[0]
#
#         for i in range(1, N):
#             stressH1 = engStress[i]
#             dstressH = stressH1 - stressH0
#             dt = time[i] - time[i - 1]
#             stress[i] = stressH1
#
#             for j in range(len(g)):
#                 stressV[j] = (
#                         np.exp(-dt / tau[i]) * stressV[j] +
#                         g[j] * stressH0 * (1 - np.exp(-dt / tau[i])) +
#                         g[j] * dstressH / dt * (dt - tau[i] + tau[i] * np.exp(-dt / tau[i]))
#                 )
#                 stress[i] -= stressV[j]
#
#             stressH0 = stressH1
#
#
#     return stress

# def LVE_expr(time, strain, sed1, sed2):
#
#     N = len(time)
#     # time = time[:,0]
#     # N = 100
#     stress = np.zeros(N)
#
#     # g = [0.1252, 0.19]
#     # tau = [110.85, 1068]
#
#     # g = [0.19]
#     # tau = [500]
#
#     # g = [0.0577, 0.173, 0.108]
#     # tau = [42.45, 15260, 221.9]
#     # stressV = np.zeros(len(g))
#     stressV = np.zeros(N)
#     Is, stretch = getIs(strain)
#     tau = sed1(Is)
#     tau = 500 + tau
#     g = sed2(Is)
#
#     W = W_expr(Is)
#     I1 = Is[:, 0]
#     I2 = Is[:, 1]
#
#     dW_dI1 = np.gradient(W, I1, edge_order=2)
#     dW_dI2 = np.gradient(W, I2, edge_order=2)
#
#     stress_all = predictStress(dW_dI1, dW_dI2, stretch, Is)
#
#     engStrain, engStress = ConvertTrueToEng(strain, stress_all)
#
#     stressH0 = engStress[0]
#
#     for i in range(1, N):
#         stressH1 = engStress[i]
#         dstressH = stressH1 - stressH0
#         dt = time[i] - time[i - 1]
#         stress[i] = stressH1
#
#         for j in range(len(g)):
#             stressV[j] = (
#                     np.exp(-dt / tau[i]) * stressV[j] +
#                     g[j] * stressH0 * (1 - np.exp(-dt / tau[i])) +
#                     g[j] * dstressH / dt * (dt - tau[i] + tau[i] * np.exp(-dt / tau[i]))
#             )
#             stress[i] -= stressV[j]
#
#         stressH0 = stressH1
#
#     return stress


# def LVE_expr(time, strain, sed_list):
#     """
#     Compute viscoelastic stress contributions from multiple SED models (sed_list).
#
#     Parameters
#     ----------
#     time : ndarray
#         Time array (shape N).
#     strain : ndarray
#         Strain array (shape N).
#     sed_list : list of callables
#         Each sed is a symbolic expression that returns tau(Is).
#
#     Returns
#     -------
#     stress_total : ndarray
#         Combined stress response from all tau contributions.
#     """
#     N = len(time)
#     stress_total = np.zeros(N)
#
#     # loop over each sed expression
#     for sed in sed_list:
#         stress = np.zeros(N)
#         stressV = np.zeros(1)  # assuming one g, can generalize if needed
#         g = [0.19]  # relaxation coefficient(s)
#
#         # invariants and stretch
#         Is, stretch = getIs(strain)
#         # tau = sed(Is)
#         # tau = np.clip(500 + tau, 1e-6, None)  # avoid negatives/zeros
#         tau1=sed_list[0](Is)
#         tau2=sed_list[1](Is)
#         tau_list = [tau1, tau2]
#         # strain energy W and derivatives
#         W = W_expr(Is)
#         I1 = Is[:, 0]
#         I2 = Is[:, 1]
#         dW_dI1 = np.gradient(W, I1, edge_order=2)
#         dW_dI2 = np.gradient(W, I2, edge_order=2)
#
#         # base elastic stress
#         stress_all = predictStress(dW_dI1, dW_dI2, stretch, Is)
#         engStrain, engStress = ConvertTrueToEng(strain, stress_all)
#
#         stressH0 = engStress[0]
#
#         # time stepping
#         for i in range(1, N):
#             stressH1 = engStress[i]
#             dstressH = stressH1 - stressH0
#             dt = time[i] - time[i - 1]
#             stress[i] = stressH1
#
#             for j in range(len(g)):
#                 tau_j = tau_list[j]
#
#                 stressV[j] = (
#                         np.exp(-dt / tau_j) * stressV[j]
#                         + g[j] * stressH0 * (1 - np.exp(-dt / tau_j))
#                         + g[j] * dstressH / dt * (dt - tau_j + tau_j * np.exp(-dt / tau_j))
#                 )
#                 stress[i] -= stressV[j]
#
#             stressH0 = stressH1
#
#         # accumulate contribution from this sed
#         stress_total += stress
#
#     return stress_total


#
#
#
#
# # Завантаження експерименту
# file_path = "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation5.TRA"
# # file_path = "D:/exper_data/TPU Experiments/TensileStrength/TensileTPU-2 strength5.TRA"
# data = np.genfromtxt(file_path, delimiter=';', dtype=None, encoding='utf-8', skip_header=8)
#
# time_raw = data[1:, 0]
# strain_raw = data[1:, 2]/100
# stress_raw = data[1:, 3]
#
# strain_max = strain_raw.max()
#
# # t_rise = 100
# # strain_phys_time = time_raw.copy()
# # strain_phys = np.piecewise(
# #     strain_phys_time,
# #     [strain_phys_time <= t_rise, strain_phys_time > t_rise],
# #     [lambda t: (strain_max / t_rise) * t, strain_max]
# # )
#
# # t_rise = 50
# t_rise = time_raw[np.argmax(strain_raw)]
# t_total = time_raw.max()
# extra_strain = 0.001 * strain_max  # приріст на 0,1% після t_rise
#
# # Побудова strain
# strain_phys_time = time_raw.copy()
# strain_phys = np.piecewise(
#     strain_phys_time,
#     [strain_phys_time <= t_rise, strain_phys_time > t_rise],
#     [
#         lambda t: (strain_max / t_rise) * t,
#         lambda t: strain_max + extra_strain * (t - t_rise) / (t_total - t_rise)
#     ]
# )
# #Список файлів
# file_paths = [
#     "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation1.TRA",
#     "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation4.TRA",
#     "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation7.TRA",
#     "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation10.TRA",
#     # "D:/exper_data/TPU Experiments/Relaxation/TensileTPURelaxation13.TRA"
# ]
#
# # 🔁 Проходимо по кожному файлу
# for i, file_path in enumerate(file_paths):
#     # Завантаження експерименту
#     data = np.genfromtxt(file_path, delimiter=';', dtype=None, encoding='utf-8', skip_header=8)
#
#     time_raw = data[1:, 0]
#     strain_raw = data[1:, 2] / 100
#     stress_raw = data[1:, 3]
#
#     strain_max = strain_raw.max()
#     t_rise = time_raw[np.argmax(strain_raw)]
#     t_total = time_raw.max()
#     extra_strain = 0.001 * strain_max  # +0.1% зростання після t_rise
#
#     # Побудова strain
#     strain_phys_time = time_raw.copy()
#     strain_phys = np.piecewise(
#         strain_phys_time,
#         [strain_phys_time <= t_rise, strain_phys_time > t_rise],
#         [
#             lambda t: (strain_max / t_rise) * t,
#             lambda t: strain_max + extra_strain * (t - t_rise) / (t_total - t_rise)
#         ]
#     )
#
#     # Інтерполяція
#     N = 100
#     time = np.linspace(time_raw.min(), time_raw.max(), N)
#     strain = np.interp(time, strain_phys_time, strain_phys)
#     stress_exp = np.interp(time, time_raw, stress_raw)
#
#     # Розрахунок моделі
#     stress_model = LVE_expr(time, strain)
#
#     # Додати графік
#     plt.plot(time, stress_model, label=f'Модель {i+1}', linestyle='-')
#     plt.scatter(time, stress_exp, label=f'Експеримент {i+1}', s=15)
#
# # Параметри графіка
# plt.xlabel('Час [с]')
# plt.ylabel('Напруження [MPa]')
# plt.legend()
# plt.grid(True)
# plt.title('Порівняння 5 експериментів з моделлю')
# plt.show()

def LVE_expr(time, strain, sed):
    N = len(time)

    stress = np.zeros(N)

    # g = [0.1252, 0.19]
    # tau = [110.85, 1068]
    g = [0.19]
    tau = [500]

    # g = [0.0577, 0.173, 0.108]
    # tau = [42.45, 15260, 221.9]
    stressV = np.zeros(len(g))

    Is, stretch = getIs(strain)

    W = W_expr(Is)
    I1 = Is[:, 0]
    I2 = Is[:, 1]

    dW_dI1 = np.gradient(W, I1, edge_order=2)
    dW_dI2 = np.gradient(W, I2, edge_order=2)

    stress_all = predictStress(dW_dI1, dW_dI2, stretch, Is)

    engStrain, engStress = ConvertTrueToEng(strain, stress_all)

    stressH0 = engStress[0]

    for i in range(1, N):
        stressH1 = engStress[i]
        dstressH = stressH1 - stressH0
        dt = time[i] - time[i - 1]
        stress[i] = stressH1

        for j in range(len(g)):
            stressV[j] = (
                    np.exp(-dt / tau[j]) * stressV[j] +
                    g[j] * stressH0 * (1 - np.exp(-dt / tau[j])) +
                    g[j] * dstressH / dt * (dt - tau[j] + tau[j] * np.exp(-dt / tau[j]))
            )
            stress[i] -= stressV[j]

        stressH0 = stressH1


    return stress



nakjjk