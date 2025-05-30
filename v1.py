#%% 

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random as ra
import os

results_dir = "results"
os.makedirs(results_dir, exist_ok = True)

print("Importy gotowe")


#%%
print("Elementy modelu: J1, J2, k, b, n1, n2.")
print("\n")
print("Aby wylosowac wartosci kliknij 1, aby wprowadzic recznie kliknij 2")

wart = 0

while wart != 1 and wart != 2:
    wart = int(input("Wartosc = "))

    if wart == 1:
        J1 = ra.uniform(0.005, 0.05)
        J2 = ra.uniform(0.01, 0.1)
        k = ra.uniform(5, 50)
        b = ra.uniform(0.01, 1.0)
        n1 = 1
        n2 = ra.randint(1, 5)
        
        print(f"J1 = {J1}, J2 = {J2}, k = {k}, b = {b}, n1 = {n1}, n2 = {n2}")
    
    if wart == 2:
        J1 = float(input("J1 = "))
        J2 = float(input("J2 = "))
        k = float(input("k = "))
        b = float(input("b = "))
        n1 = int(input("n1 = "))
        n2 = int(input("n2 = "))
        
        print(f"J1 = {J1}, J2 = {J2}, k = {k}, b = {b}, n1 = {n1}, n2 = {n2}")
        
def model(t, x, Tm):
    dx1 = x[1]
    dx2 = (Tm*n2/n1 - b* x[1] - k*n2/n1* x[0]) / (J1*(n2/n1)**2 + J2)
    
    return np.array([dx1, dx2])


#%%


    

    

