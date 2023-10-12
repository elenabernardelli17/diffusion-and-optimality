# import numpy as np
# import matplotlib.pyplot as plt
# # Parametro del problema
# D = 1.3e-4*3600  # Costante di diffusione

# # Range di valori di x
# x = np.linspace(-10, 10, 400)

# # Valori di tempo per cui vuoi plottare P(x)
# t_values = [4,8,12,24,36,48]

# plt.figure(figsize=(6, 4))

# # Calcola e plotta la distribuzione di probabilità P(x) per ciascun valore di tempo
# for t in t_values:
#     P_x = 1.0 / np.sqrt(4 * np.pi * D * t) * np.exp(-x**2 / (4 * D * t))
#     plt.plot(x, P_x, label=f't={t} hours')

# plt.title(f'Probability distribution of P(x) at different time values')
# plt.xlabel('space [mm]')
# plt.ylabel('P(x)')
# # plt.grid(True)
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Definisci i valori di x e t su cui desideri creare il contour plot
x = np.linspace(0, 10, 100)
# t = np.linspace(0.1, 48, 0.1)
t = np.arange(0.1, 48, 0.1)

# Crea una griglia di punti x e t
X, T = np.meshgrid(x, t)

# Calcola i valori di C tilde utilizzando l'equazione data
D = 1.3e-4*3600  # Puoi regolare il valore di D se necessario
C_tilde = erfc(X / (2 * np.sqrt(D * T)))

# Crea il contour plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, T, C_tilde, levels=20, cmap='viridis')
plt.colorbar(label=r'Concentration [mm$^{-3}$]')
plt.xlabel(r'Distance [mm]')
plt.ylabel(r'Time [h]')
plt.title(r'Solution in Time and Space of $\tilde{C}(x,t)$')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Definisci i valori di x e t su cui desideri creare il contour plot
x = np.linspace(0, 10, 100)
# t = np.linspace(0.1, 48, 200)
t = np.arange(0.1, 48, 0.1)

# Crea una griglia di punti x e t
X, T = np.meshgrid(x, t)

# Calcola i valori di HatC utilizzando l'equazione data
D = 1.3e-4*3600  # Puoi regolare il valore di D se necessario
C_tilde = erfc(X / (2 * np.sqrt(D * T)))
HatC = C_tilde * T * (48 - T)

# Crea il contour plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, T, HatC, levels=20, cmap='viridis')
plt.colorbar(label=r'Concentration [mm$^{-3}$]')
plt.xlabel(r'Distance [mm]')
plt.ylabel(r'Time [h]')
plt.title(r'Solution in Time and Space of $\hat{C}(x,t)$')
plt.show()
