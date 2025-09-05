import numpy as np
# from dso.DataWrapper import *
# from dso.program import Program
import lib_modification.DataWrapper
from scipy import integrate
import sympy as sp
from lib_modification.cont_m import *


def stress(lam1, SED):
    W = SED(lam1)
    h = 0.001
    sigma = np.zeros(lam1.size)

    lam1_h = np.copy(lam1) + h
    W_direction = SED(lam1_h)
    dw_dl = (W_direction - W) / h

    lam1 = lam1.ravel()
    sigma = lam1 * dw_dl
    y_pred = sigma
    return y_pred

# def strain_to_lam(trueStrain):
#     lam = np.zeros((trueStrain.size, 3))
#     lam[:, 0] = trueStrain + np.ones(trueStrain.shape)
#     lam[:, 1] = 1 / np.sqrt(lam[:, 0])
#     lam[:, 2] = 1 / np.sqrt(lam[:, 0])
#     return lam


def principal_cauchy_stress(lam, sed):
    # lam = strain_to_lam(X)
    W = sed(lam)
    h = 0.001
    l_i, l_j = lam.shape
    directions = np.eye(3)
    dw_dl = np.zeros((l_i, 3))

    for i in range(0, 3):
        lam_h = np.copy(lam) + h*directions[i, :]
        W_direction = sed(lam_h)
        dw_dl[:, i] = (W_direction - W) / h

        # lam[:, i] = lam[:, i].ravel()
    sigma = lam[:, 0] * dw_dl[:, 0] - lam[:, 1] * dw_dl[:, 1]
    y_pred = sigma
    return y_pred


# def central_diff_first_derivative(y, x):
#     h = x[1] - x[0]
#     dydx = (y[2:] - y[:-2]) / (2 * h)
#     # Pad the result to match the length of the input array
#     dydx = np.pad(dydx, (1, 1), 'edge')
#     return dydx
#
# def central_diff_second_derivative(y, x):
#     h = x[1] - x[0]
#     d2ydx2 = (y[2:] - 2 * y[1:-1] + y[:-2]) / h**2
#     # Pad the result to match the length of the input array
#     d2ydx2 = np.pad(d2ydx2, (1, 1), 'edge')
#     return d2ydx2

# def gauss_integrate(f, a, b, n):
#     x, w = np.polynomial.legendre.leggauss(n)
#     # Change of interval from [-1, 1] to [a, b]
#     t = 0.5 * (x + 1) * (b - a) + a
#     return 0.5 * (b - a) * np.sum(w * f)


def Stress_visco(tau, strain_rate_history, d2wd2e, E1, eta, eps, G2=None):
    # sum = d2wd2e + (G2 if G2 is not None else 0)
    sum = d2wd2e*eps
    subIntegral = sum * strain_rate_history
    stress = np.trapz(subIntegral, tau)
    # stress = integrate.simps(subIntegral, x=tau)
    # stress = np.real(np.trapz(subIntegral, tau))
    # stress = gauss_integrate(subIntegral, 0, tau, 2)

    # check with symbolic calculation
    # if isinstance(stress, complex):
    #     print("Value is complex.")
    #     print('stress =', stress)

    return stress


# def CallExecute(eps, sed, DataWrapper):
#
#     if DataWrapper.FitStrainEnergy == True:
#         sigma = stress_MR(eps, sed, DataWrapper)
#     elif DataWrapper.FitViscous == True:
#         sigma = stress_MR_int(eps, sed, DataWrapper)
#
#     return sigma


# def CallExecute(Is, sed, DataWrapper):
#
#     # W = sed(Is)
#     # dPsidI1 =np.gradient(W, Is[:, 0], edge_order=2)
#     # dPsidI2 =np.gradient(W, Is[:, 1], edge_order=2)
#     # stretch = DataWrapper.stretch
#     # modelresponse = predictStress(dPsidI1, dPsidI2, stretch, Is)
#     strain = DataWrapper.strain
#     if DataWrapper.FitStrainEnergy == True:
#         sigma = predictStress(dPsidI1, dPsidI2, stretch, Is)
#     elif DataWrapper.FitViscous == True:
#         sigma = LVE_expr(Is, strain, sed)
#         # sigma = LVE_expr(Is, sed, DataWrapper)
#     return sigma



def CallExecute(Is, sed, DataWrapper):


    # modelresponse = predictStress(dPsidI1, dPsidI2, stretch, Is)
    # time = DataWrapper.time
    strain = DataWrapper.strain
    # strain = Is[:,1]
    time = Is[:,0]
    if DataWrapper.FitStrainEnergy == True:
        W = sed(Is)
        dPsidI1 =np.gradient(W, Is[:, 0], edge_order=2)
        dPsidI2 =np.gradient(W, Is[:, 1], edge_order=2)
        stretch = DataWrapper.stretch
        sigma = predictStress(dPsidI1, dPsidI2, stretch, Is)
    elif DataWrapper.FitViscous == True:
        sigma = LVE_expr(time, strain, sed)
        # sigma = LVE_expr(Is, time,sed, DataWrapper)
        # sigma = LVE_expr(Is, sed, DataWrapper)
    return sigma


# def CallExecute(Is, sed_list, DataWrapper):
#     """
#     Run stress computation with multiple tau arrays in parallel.
#
#     Parameters
#     ----------
#     Is : ndarray
#         Invariants (input features).
#     sed_list : list of callables
#         Each element is a symbolic expression (sed) returning tau(Is).
#     DataWrapper : object
#         Contains strain, stretch, time arrays.
#     """
#     strain = Is[:, 1]
#     time = Is[:,0]
#     stress_total = np.zeros_like(DataWrapper.strain)
#
#     # loop over each symbolic discovery expression
#     for sed in sed_list:
#         tau = sed(Is)  # tau array for this mode
#         tau = np.clip(tau, 1e-6, None)  # protect against negatives/NaN
#
#         # compute contribution from this tau
#         if DataWrapper.FitStrainEnergy == True:
#             sigma = predictStress(dPsidI1, dPsidI2, stretch, Is)
#         elif DataWrapper.FitViscous == True:
#             stress_total = LVE_expr(time, strain, sed_list)
#             # stress_total += contribution
#
#     return stress_total

def stress_MR(eps, sed, DataWrapper):
    t = DataWrapper.time
    E1 = DataWrapper.E
    eta = DataWrapper.etha
    t = t.ravel()

    W = sed(eps)

    dt = t[1] - t[0]
    t1 = t[1]
    vect_len = t.size
    eps = eps.ravel()
    sigma_vis_el = np.zeros(vect_len)

    # d2w = np.gradient(sigma, eps)
    # d2 = central_diff_first_derivative(W, eps)
    # d2Wd2e = central_diff_second_derivative(W, eps)

    #check with symbolic calculation
    #d2w = sp.diff(Program, sp.symbols('x1')) #np.gradient(W, eps, edge_order=2)
    #d2Wd2e = sp.diff(Program, sp.symbols('x1'))

    d2w = np.gradient(W, eps, edge_order=2)
    d2Wd2e = np.gradient(d2w, eps, edge_order=2)

    # # Видаляємо останні 2 точки
    # d2Wd2e_new = d2Wd2e[:-2]
    #
    # # Вибираємо дві попередні точки2
    # x1, x2 = d2Wd2e_new[-2], d2Wd2e_new[-1]
    #
    # # Визначаємо індекси цих точок
    # idx1, idx2 = len(d2Wd2e_new) - 2, len(d2Wd2e_new) - 1
    #
    # # Визначаємо уявну пряму через ці точки
    # slope = (x2 - x1) / (idx2 - idx1)
    # intercept = x1 - slope * idx1
    #
    # # Обчислюємо значення для двох видалених точок на основі прямої
    # new_x1 = slope * len(d2Wd2e_new) + intercept
    # new_x2 = slope * (len(d2Wd2e_new) + 1) + intercept
    #
    # # Додаємо ці значення до масиву
    # d2Wd2e_new = np.append(d2Wd2e_new, [new_x1, new_x2])
    # d2Wd2e = d2Wd2e_new
    # d2Wd2e = sp.lambdify((sp.symbols("x1"), d2Wd2e, modules='numpy'))

    eps_rate = np.gradient(eps, t, edge_order=2)


    for i in range(vect_len):
        #print('second derivative =', d2Wd2e.subs(sp.symbols('x1'), eps[i]).evalf())
        # sigma_vis_el[i] = Stress_visco(t[t < t1], eps_rate[:t[t < t1].size], sp.re(d2Wd2e.subs(sp.symbols('x1'), eps[i])).evalf(), E1, eta)
        sigma_vis_el[i] = Stress_visco(t[t < t1], eps_rate[:t[t < t1].size], d2Wd2e[i], E1, eta, eps[i])
        t1 = t1 + dt

    return sigma_vis_el

def stress_MR_int(X, sed, DataWrapper):
    # t = X[:, 0]
    # eps = X[:, 1]
    t = X
    t = t.ravel()
    eps = DataWrapper.strain
    E1 = DataWrapper.E
    eta = DataWrapper.etha

    vect_len = t.size
    eps_rate = np.gradient(eps, t)
    dt = (t[1] - t[0])/100
    t1 = t[1]


    sigma_vis_el = np.zeros(vect_len)
    # d = eps**2*(eps*(1.909053e-6*eps - 0.000468068) + 0.0851525961)
    # d = np.sqrt(eps**2 * (eps*(0.0005446501157938892*eps + 0.70489051458157166)+0.8764146566052))

    # d = eps**2 * (-6.722799290036608e-5 * eps *(0.9992580914668002 - 0.0008366490180128791*eps)+ 0.056090498119589056)


    eps_safe = np.where(eps > 0, eps, 1e-10)
    d2 = np.gradient(d, eps_safe)

    d2Wd2e = np.gradient(d2, eps)
    #
    # # Видаляємо останні 2 точки
    # d2Wd2e_new = d2Wd2e[:-2]
    #
    # # Вибираємо дві попередні точки
    # x1, x2 = d2Wd2e_new[-2], d2Wd2e_new[-1]
    #
    # # Визначаємо індекси цих точок
    # idx1, idx2 = len(d2Wd2e_new) - 2, len(d2Wd2e_new) - 1
    #
    # # Визначаємо уявну пряму через ці точки
    # slope = (x2 - x1) / (idx2 - idx1)
    # intercept = x1 - slope * idx1
    #
    # # Обчислюємо значення для двох видалених точок на основі прямої
    # new_x1 = slope * len(d2Wd2e_new) + intercept
    # new_x2 = slope * (len(d2Wd2e_new) + 1) + intercept
    #
    # # Додаємо ці значення до масиву
    # d2Wd2e_new = np.append(d2Wd2e_new, [new_x1, new_x2])
    # d2Wd2e = d2Wd2e_new

    # Stress_visco(t[t<t1], sed(t[t<t1]))
    for i in range(vect_len):
        # G_visc = sed(t1 - t)
        # G = G_visc + d2Wd2e
        Tt = t1 - t
        Tt = np.array(Tt)
        # combined_array = np.column_stack((Tt, eps))

        SED = sed(Tt.reshape(-1, 1))
        # SED = sed(combined_array)

        # SED = -0.0010997865234680451 * np.exp(8*Tt**4 * (1.4454 - Tt))
        # ex = 0.1*np.exp(-0.1*Tt/eta)

        sigma_vis_el[i] = Stress_visco(t[t < t1], eps_rate[:t[t < t1].size], d2Wd2e[i], E1, eta, SED[:t[t < t1].size])
        t1 = t1 + dt
    return sigma_vis_el

def stress_MR_spec(t, sed, DataWrapper):
    t = t.ravel()
    eps = DataWrapper.strain
    E1 = DataWrapper.E
    eta = DataWrapper.etha

    vect_len = t.size
    eps_rate = np.gradient(eps, t)
    dt = t[1] - t[0]
    t1 = t[1]


    # sigma = (c01 + c10 / lam) * (lam * lam - 1 / lam)
    # d2Wd2e = np.gradient(sigma, eps)
    # d = -12.65820147614499 * np.power(eps, 2) * (-eps - 1.604679235407087) * (np.power(eps, 2) - 4.211133416484)
    # d = eps*(-eps**2 + 4*eps) + eps
    # for i in range(len(d2Wd2e)):
    #     if np.isnan(d2Wd2e[i]):
    #         # print(d2w[i])
    #         d2Wd2e[i] = d2Wd2e[i - 1]

            #
    sigma_vis_el = np.zeros(vect_len)
    # d = (1/2)*E1*np.power(eps,2)
    # d = 60.24626324213107*np.power(eps,2)*(9.488588273963604*np.power(eps,2)+8.941642543303079)
    # d = 35.24991028831693 * np.power(eps, 3) * (1.5977850872318107 * eps + 4.84219393296361)

    d = 107.83643130563685 * eps * (-np.power(eps, 3)*(-eps - 2.134540607352205) + eps)
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

    z = np.linspace(0, t[-1]*2, 100)
    z = z.reshape(-1, 1)

    SED = sed(z) # дельта функція
    z=z.ravel()
    # Stress_visco(t[t<t1], sed(t[t<t1]))
    for i in range(vect_len):
        # G_visc = sed(t1 - t)
        # G = G_visc + d2Wd2e
        Tt = t1 - t
        # ex = 13.99317635356684 - 1761.356446696241*Tt
        G2 = np.zeros(len(Tt))
        for j in range(len(Tt)):
            G_subi = SED*np.exp(-Tt[j]/z)
            G2[j] = np.trapz(G_subi, z)
        # ex = 0.1*np.exp(-0.1*Tt/eta)
        sigma_vis_el[i] = Stress_visco(t[t < t1], eps_rate[:t[t < t1].size], d2Wd2e[i], E1, eta, G2[:t[t < t1].size])
        t1 = t1 + dt
    return sigma_vis_el

