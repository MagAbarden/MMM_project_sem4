import numpy as np
import matplotlib.pyplot as plt

# Parametry układu (możliwe do zmiany)
k = 10.0    # sztywność sprężyny
b = 1.0     # tłumienie
J1 = 1.0    # moment bezwładności 1
J2 = 2.0    # moment bezwładności 2

# Funkcje generujące sygnały wejściowe
def input_signal(t, type='rect', amplitude=1.0, frequency=1.0, duration=2.0):
    if type == 'rect':
        return amplitude if t < duration else 0
    elif type == 'tri':
        return amplitude * (1 - abs((t % (1/frequency)) * frequency - 0.5) * 2)
    elif type == 'sin':
        return amplitude * np.sin(2 * np.pi * frequency * t)
    else:
        return 0

# Funkcja opisująca układ (model równań)
def system(t, x, input_type):
    Tm = input_signal(t, type=input_type)
    dxdt = np.zeros_like(x)
    dxdt[0] = x[1]
    dxdt[1] = (Tm - k*(x[0] - x[2])) / J1
    dxdt[2] = x[3]
    dxdt[3] = (k*(x[0] - x[2]) - b*x[3]) / J2
    return dxdt

# Metoda Eulera
def euler_step(f, t, x, dt, input_type):
    return x + dt * f(t, x, input_type)

# Metoda Rungego-Kutty 4 rzędu
def rk4_step(f, t, x, dt, input_type):
    k1 = f(t, x, input_type)
    k2 = f(t + dt/2, x + dt/2 * k1, input_type)
    k3 = f(t + dt/2, x + dt/2 * k2, input_type)
    k4 = f(t + dt, x + dt * k3, input_type)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# Symulacja
def simulate(method='euler', input_type='rect', t_max=10, dt=0.01):
    steps = int(t_max / dt)
    t = np.linspace(0, t_max, steps)
    x = np.zeros((steps, 4))  # stan [theta1, omega1, theta2, omega2]
    for i in range(steps - 1):
        if method == 'euler':
            x[i+1] = euler_step(system, t[i], x[i], dt, input_type)
        elif method == 'rk4':
            x[i+1] = rk4_step(system, t[i], x[i], dt, input_type)
    return t, x

# Wykres wyników
def plot_results(t, x_euler, x_rk4):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2,1,1)
    plt.plot(t, x_euler[:,2], label='Theta2 (Euler)')
    plt.plot(t, x_rk4[:,2], label='Theta2 (RK4)', linestyle='--')
    plt.title('Pozycja (Theta2)')
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja [rad]')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(t, x_euler[:,3], label='Omega2 (Euler)')
    plt.plot(t, x_rk4[:,3], label='Omega2 (RK4)', linestyle='--')
    plt.title('Prędkość (Omega2)')
    plt.xlabel('Czas [s]')
    plt.ylabel('Prędkość [rad/s]')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- Główne wykonanie ---
input_type = 'rect'  # 'rect', 'tri', 'sin'
t_max = 10
dt = 0.001

t, x_euler = simulate(method='euler', input_type=input_type, t_max=t_max, dt=dt)
_, x_rk4 = simulate(method='rk4', input_type=input_type, t_max=t_max, dt=dt)

plot_results(t, x_euler, x_rk4)