"""
We consider the following problem

dC/dt = div(D*grad(C)) in Omega

Omega = (0, L)    L = 1 cm
D = 1.3*10**-4 mm^2/s

We want to know C after 8, 24 and 48 hours

Initial condition
C(x,0) = 0 in Omega

Buondary conditions
- Dirichlet buondary condition
C(0,t) = f(t)
f(t) = t*(48-t)

- Neumann boundary condition
-D*grad(C)*n = 0 in x = L 
"""

from __future__ import print_function
from fenics import *
from dolfin_adjoint import *
import numpy as np

import ipyopt
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import erfc
from mpl_toolkits.mplot3d import Axes3D

def anal_sol(x,t):
     
    return erfc(x/(2*np.sqrt(1.3e-4*3600*t)))


# Function to solve my diffusion problem 
def solve_dif(T, dt, N, D):
    
    # Define the domain in mm
    mesh = IntervalMesh(N, 0, 10)

    # Define time discretization parameter
    num_steps = int(T/dt)

    # Define the finite element spaces and functions
    V = FunctionSpace(mesh, "CG", 1)
    C = TrialFunction(V)
    v = TestFunction(V)         

    # Define Dirichlet buondary condition depending on time
    # time to be modified time based on the time discretization
    C_D = Expression('t*(48 - t)', t=0, degree=1)        

    # Define the left boundary domain
    def GammaL(x, on_boundary):
            return near(x[0], 0)

    # Define the Dirichlet condition 
    bc = DirichletBC(V, C_D, GammaL) 

    # Define initial value
    C_n = Function(V) 

    # d = Function(V, name="data")

    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
    a, L = lhs(F), rhs(F)

    # Solution at final time T
    C_T = Function(V)

    t = 0
    for n in tqdm(range(num_steps)):

        # Update current time
        t += dt

        # Update current Dirichlet condition
        C_D.t = t 
        
        # Update current data = analytic solution
        # data.t = t
        # d.assign(interpolate(data, V))
                
        # Compute solution solving the variational problem
        solve(a == L, C_T, bc)
        
        # Assign the ne initial value for next time step
        C_n.assign(C_T)
    
    # This function returns the solution at final time T
    return C_T

C_T = solve_dif(8, 0.1, 100, 1.3e-4*3600)

# Plot 1
# Plot with different spatial intervals
# for k in range(3): 
#     N = 10*(2**k)                    
#     print ("solving for ", N)
#     C_T = solve_dif(1, 0.1, N, 1.3e-4*3600)
#     V = C_T.function_space()
#     plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), linewidth=3)

# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Concentration [mm$^{{-3}}$]')
# legend = [f"N={10*(2**k)}" for k in range(3)]
# plt.legend(legend, loc = 1)
# plt.title("Solution of Diffusion Problem after 48 hours, $\Delta t = 3.6$ seconds")
# plt.show()

space = np.linspace(0, 10, 100)  
time = np.linspace(0, 8, int(0.1))

# # Plot 2
# # Initialize an empty list to store the curves
# concentration_curves = []

# # Calculate concentration curves for various time points
# for t in time:
#     C_T = solve_dif(t, 1, 100, 1.3e-4*3600)
#     concentration_values = [C_T(x) for x in space]
#     concentration_curves.append(concentration_values)

# # Plot concentration curves with different colors
# for i, curve in enumerate(concentration_curves):
#     plt.plot(space, curve, label=f'Time: {time[i]:.1f}')

# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Concentration [mm$^{{-3}}$]')
# # plt.legend()
# plt.title('Concentration Evolution Over Time')
# plt.show()

# # Plot 3
# # Initialize a matrix to store concentration values
# concentration_matrix = np.zeros((len(time), len(space)))

# Calculate concentration values for different points in time and space
# for i, t in enumerate(time):
#     C_T = solve_dif(t, 1, 100, 1.3e-4*3600)
#     for j, x in enumerate(space):
#         concentration_matrix[i, j] = C_T(x)

# # Create a heatmap-like plot to represent diffusion
# plt.imshow(concentration_matrix, extent=[space.min(), space.max(), time.min(), time.max()],
#            aspect='auto', cmap='viridis', origin='lower')

# # Add vertical dividing lines
# num_vertical_lines = 5  # Adjust the number of vertical lines as needed
# space_dividers = np.linspace(space.min(), space.max(), num_vertical_lines + 1)
# for space_divider in space_dividers[1:-1]:
#     plt.axvline(x=space_divider, color='white', linestyle='--', linewidth=1)
# plt.colorbar(label=r'Concentration [mm$^{{-3}}$]')
# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Time [h]')
# plt.title('Diffusion in Time and Space')
# plt.show()

# # Plot 4
# Create a filled contour plot to represent diffusion
# plt.contourf(space, time, concentration_matrix, cmap='viridis', levels=20)
# plt.colorbar(label=r'Concentration [mm$^{{-3}}$]')
# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Time [h]')
# plt.title('Diffusion in Time and Space')
# plt.show()



# plt.figure()
# C_T = solve_dif(48*60, 1, 100, 1.3e-4*60)
# V = C_T.function_space()
# plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), color ='red', label = 'Numerical Solution')
# space = np.arange(0, 10, 0.1)
# print(anal_sol(space, 48*60 - 1))
# plt.plot(space, anal_sol(space, 48*60), color = 'blue', linestyle='--', label = 'Analytic Solution')
# plt.xlabel(r'distance [mm]')
# plt.ylabel(r'concentration [mm$^{{-3}}$]')
# plt.legend()
# plt.title('Comparison between Numerical and Analytical Solution after 48 hours')
# plt.show()

# Plot the analytical solution
# Define the values of t for which you want to plot the function
t_values = [10.0]

# Create an array of x-values
x_values = np.arange(0, 10, 0.01)  # Adjust the range and number of points as needed

# Plot the function for each value of t
for t in t_values:
    C_values = erfc(x_values / (2 * np.sqrt(1.3e-4 * 3600 * t))) * (t * (48 - t))
    plt.plot(x_values, C_values, label=f't = {t} h')

plt.xlabel('Space [mm]')
plt.ylabel('Function Value [mm/h]')
plt.legend()
plt.title('Analytical solution with respect to space for different time values t')
plt.show()













