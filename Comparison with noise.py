from solve_diffusion import solve_dif
from noise_Dirichlet_condition import solve_dif_with_noise
import numpy as np
from fenics import *
import matplotlib.pyplot as plt

# Define parameters for your simulations
T = 1
dt = 0.1
N = 100
D = 1.3e-4 * 3600
noise_times = [0.5, 0.7]
noise_std_dev = 0.1

# Create the mesh and function space
mesh = IntervalMesh(N, 0, 10)
V = FunctionSpace(mesh, "CG", 1)

# Solve the diffusion problem without noise
solution_without_noise = solve_dif(T, dt, N, D)

# Solve the diffusion problem with noise
solution_with_noise = solve_dif_with_noise(T, dt, N, D, noise_times, noise_std_dev)

# Plot both solutions in the same figure
plt.figure()
plt.plot(V.tabulate_dof_coordinates(), solution_without_noise.vector().get_local(), linewidth=2, label='Without Noise')
plt.plot(V.tabulate_dof_coordinates(), solution_with_noise.vector().get_local(), linewidth=2, linestyle='--', label='With Noise')
plt.title('Comparison with and without noise on the Dirichlet condition')
plt.xlabel('Space')
plt.ylabel('Concentration')
plt.legend()
plt.show()

   