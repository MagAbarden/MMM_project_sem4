#%% IMPORTY

import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
results_dir = "results"
os.makedirs(results_dir, exist_ok = True)

print("Importy gotowe")


#%% DOPASOWANIE ELEMENTÓW MODELU
print("Elementy modelu: J1, J2, k, b, n1, n2.")
print("\n")
print("Aby wybrac przykladowe wartosci kliknij 1, aby wprowadzic recznie kliknij 2")

wart = 0
while wart != 1 and wart != 2:
    wart = int(input("Wartosc (1 albo 2)"))

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
t = 30
t_krok = np.arange(0, t, dt)  


#%% WYBÓR SYGNAŁU WEJSCIOWEGO
        
print("Wybierz sygnal wejsciowy: 1 - prostokatny, 2 - trojkatny, 3 - harmoniczny")
wart = 0
f = 1
Tm = np.zeros(len(t_krok))
pr_okres = int(1 / (f * dt))

while wart != 1 and wart != 2 and wart != 3:
    wart = int(input("Wybor: "))

    if wart == 1:
        A = float(input("Amplituda = "))
        
        for i in range(len(t_krok)):
            if (i % pr_okres) < (pr_okres // 2):
                Tm[i] = A/2
            else:
                Tm[i] = -A/2
        
    if wart == 2:
        A = float(input("Amplituda = "))
        for i in range(len(t_krok)):
            if (i % pr_okres) < (pr_okres // 2):
                Tm[i] = (2 * A / pr_okres) * (i % pr_okres) - A/2
            else:
                Tm[i] = (-2 * A / pr_okres) * (i % pr_okres - pr_okres // 2) + A/2
           
    if wart == 3:
        A = float(input("Amplituda = "))
        Tm = A * np.sin(2 * np.pi * f * t_krok)   
        
def model(x, U):
    dx1 = x[1]
    dx2 = (U*n2/n1 - b* x[1] - k*n2/n1* x[0]) / (J1*(n2/n1)**2 + J2)
    
    return np.array([dx1, dx2])

plt.title('Sygnal pobudzajacy Tm')
plt.xlabel('t [s]')
plt.ylabel('U [V]')
plt.plot(t_krok, Tm, color = 'C4')
plt.savefig("results/sygnal.png", dpi = 300)
plt.show()


#%% METODA EULERA

def euler(Tm): 
    x = [3, 0]
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

#%% METODA RK4
def rk4_step(f,t_krok,dt):
    x = [3, 0]
    rk4_1 = []
    rk4_2 = []
    for i in range(len(t_krok)):
        u = Tm[i]
        k1 = dt*f(x,u)
        k2 = dt*f(x + k1/2, u + dt/2)
        k3 = dt*f(x + k2/2, u + dt/2)
        k4 = dt*f(x + k3, u + dt)
        x[0] = x[0] + (1 / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        x[1] = x[1] + (1 / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        rk4_1.append(x[0])
        rk4_2.append(x[1])
    return rk4_1, rk4_2

rk4_1, rk4_2 = rk4_step(model, t_krok,dt)

plt.grid(True)
plt.plot(t_krok, rk4_1, color = 'C1', lw = 5, label = "RK4")
plt.plot(t_krok, euler_dx1, color = '0.0',ls = '--', label = "Euler")
plt.title('Wykres położenia ')
plt.xlabel('czas')
plt.ylabel('polożenie')
plt.legend()
plt.savefig("results/Polozenie.png", dpi=300)
plt.show()


plt.plot(t_krok, rk4_2, color = 'C1', lw = 5, label = "RK4")
plt.plot(t_krok, euler_dx2, color = '0.0', ls = '--', label = "Euler")
plt.title('Wykres prędkośći')
plt.xlabel('czas')
plt.ylabel('prędkość')
plt.legend()
plt.savefig("results/Predkosc.png", dpi=300)
plt.show()  


plt.plot(t_krok, rk4_1, color = '#7ec265', label = "położenie")
plt.plot(t_krok, rk4_2, color = "#d9687b", ls = ':', label = "prędkość")
plt.title('RK4')
plt.xlabel('czas')
plt.ylabel('wartość')
plt.legend()
plt.savefig("results/RK4.png", dpi=300)
plt.show()


plt.plot(t_krok, euler_dx1, color = "#20f1f1", label = "położenie")
plt.plot(t_krok, euler_dx2, color = "#4b0909", ls = ':', label = "prędkość")
plt.title('Euler')
plt.xlabel('czas')
plt.ylabel('wartość')
plt.legend()
plt.savefig("results/Euler.png", dpi=300)
plt.show()

