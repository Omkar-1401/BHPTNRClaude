import numpy as np

def x(M, f):
    return (np.pi * M * f) ** (2/3) 

def dEdt(nu, x):
    return 32 * nu ** 2 * x ** 5 * (1 - (1247 / 336 + 35 / 12 * nu) * x + 4 * np.pi * x ** (3/2) + (-44711 / 9072 + 9271 / 504 * nu + 65 / 18 * nu ** 2) * x ** 2)

def dJdt(M, x, nu):
    return (32/5) * M * x**(7/2) * nu**2 * (1+ x * (-(1247/336) - (35*nu)/12)+ + 4 * np.pi * x ** (3/2) + x**2 * (-(44711/9072) + (9271*nu)/504 + (65*nu**2)/18))