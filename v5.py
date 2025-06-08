#%% 

import numpy as np
import matplotlib.pyplot as plt
import os

results_dir = "results"
os.makedirs(results_dir, exist_ok = True)

print("Importy gotowe")


#%%
print("Elementy modelu: J1, J2, k, b, n1, n2.")
print("\n")
print("Aby wybrac przykladowe wartosci kliknij 1, aby wprowadzic recznie kliknij 2")

wart = 0
while wart != 1 and wart != 2:
    wart = int(input("Wartosc = "))

    if wart == 1:
        
        J1 = 0.05
        J2 = 0.1
        k = 10
        b = 1
        n1 = 1
        n2 = 2
        
        print(f"J1 = {J1}, J2 = {J2}, k = {k}, b = {b}, n1 = {n1}, n2 = {n2}")
    
    if wart == 2:
        J1 = float(input("J1 = "))
        J2 = float(input("J2 = "))
        k = float(input("k = "))
        b = float(input("b = "))
        n1 = int(input("n1 = "))
        n2 = int(input("n2 = "))
        
        print(f"J1 = {J1}, J2 = {J2}, k = {k}, b = {b}, n1 = {n1}, n2 = {n2}")
        
        
dt = 0.001
t = 10
t_krok = np.arange(0, t, dt)  


#%%
        
wart = 0
print("Wybierz sygnal wejsciowy: 1 - prostokatny, 2 - trojkatny, 3 - harmoniczny")
wart = 0
f = 1
Tm = np.zeros(len(t_krok))
pr_okres = int(1 / (f * dt))

while wart != 1 and wart != 2 and wart != 3:
    wart = int(input("Wybor: "))

    if wart == 1:
        A = int(input("Amplituda = "))
        
        for i in range(len(t_krok)):
            if (i % pr_okres) < (pr_okres // 2):
                Tm[i] = A/2
            else:
                Tm[i] = -A/2
        
    if wart == 2:
        A = int(input("Amplituda = "))
        for i in range(len(t_krok)):
            if (i % pr_okres) < (pr_okres // 2):
                Tm[i] = (2 * A / pr_okres) * (i % pr_okres) - A/2
            else:
                Tm[i] = (-2 * A / pr_okres) * (i % pr_okres - pr_okres // 2) + A/2
           
    if wart == 3:
        A = int(input("Amplituda = "))
        Tm = A * np.sin(2 * np.pi * f * t_krok)
        
        
def model(x, U):
    dx1 = x[1]
    dx2 = (U*n2/n1 - b* x[1] - k*n2/n1* x[0]) / (J1*(n2/n1)**2 + J2)
    
    return np.array([dx1, dx2])

plt.title('Sygnal pobudzajacy Tm')
plt.xlabel('t [s]')
plt.ylabel('U [V]')
plt.plot(t_krok, Tm)
plt.show


#%%

def euler(Tm): 
    x = [0, 0]
    euler_dx1 = np.zeros(len(t_krok))
    euler_dx2 = np.zeros(len(t_krok))
    
    for i in range(len(t_krok)):
        u = Tm[i]
        dx1, dx2 = model(x, u)
        x[0] = x[0] + dx1 * dt
        x[1] = x[1] + dx2 * dt
        
        euler_dx1[i] = x[0]
        euler_dx2[i] = x[1]
        
    return  euler_dx1, euler_dx2

euler_dx1, euler_dx2 = euler(Tm) 


plt.title('Wykres polozenia')
plt.xlabel('t [s]')
plt.ylabel('s [m]')
plt.plot(t_krok, euler_dx1)
plt.show()

plt.title('Wykres predkosci')
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.plot(t_krok, euler_dx2)
plt.show()
    