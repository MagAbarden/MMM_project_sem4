#%% IMPORTY BIBLIOTEK
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random as ra
import os
print("IMPORTy ZAKOnCZONE")


t0 = 0
t_end = 10
dt = 0.01            #krok próbkowania
Amp = 1             #Amplituda sygnału
freq = 1           #Częstotliwość
x = np.array([0,0])
#%% GENEROWANIE SYGNAŁÓW

def sygnal_prostokątny(t0,t_end,dt,Amp,freq):
    t = np.arange(t0, t_end+dt, dt)
    Tm = Amp * signal.square(2 * np.pi * freq * t)
    return t, Tm

def sygnal_trojkatny(t0,t_end,dt,Amp,freq):
    t = np.arange(t0, t_end+dt, dt)
    Tm = Amp * signal.sawtooth(2 * np.pi * freq * t, width = 0.5)
    return t, Tm

def sygnal_harmoniczny(t0,t_end,dt,Amp,freq):
    t = np.arange(t0, t_end+dt, dt)
    Tm = Amp * np.sin(2 * np.pi * freq * t)
    return t, Tm

czas, sygnal_p = sygnal_prostokątny(t0,t_end,dt,Amp,freq)
czas, sygnal_t = sygnal_trojkatny(t0,t_end,dt,Amp,freq)
czas, sygnal_h = sygnal_harmoniczny(t0,t_end,dt,Amp,freq)
#%% MODEL UKŁADU
def model(x, sygnal_p):
    dx1 = x[1]
    dx2 = x[0]* sygnal_p
    return np.array([dx1, dx2])

#%% RYSOWANIE SYGNAŁÓW
theta = model(x, sygnal_p)
print(theta[0])
plt.plot(czas, sygnal_p, label='Prostokątny', color='red')
plt.plot(czas, sygnal_t, label='Trójkątny', color='green')
plt.plot(czas, sygnal_h, label='Sinusoidalny', color='blue', linestyle='--')

plt.title("Wiele sygnałów na jednym wykresie")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.legend()  # dodaje legendę
plt.grid(True)
plt.show()
