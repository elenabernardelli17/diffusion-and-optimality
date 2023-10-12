"""
We consider the following problem

dC/dt = div(D*grad(C)) - g(x,t) in Omega

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
-D*grad(C)*n = h(x,t) in x = L 
"""

from __future__ import print_function
from fenics import *
from dolfin_adjoint import *
import numpy as np

import ipyopt
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats import linregress
# from solve_diffusion import solve_dif
# from solve_diffusion import anal_sol


def anal_sol(x,t):
     
    return erfc(x/(2*np.sqrt(1.3e-4*3600*t)))

def anal_sol_2(x,t):
     
    return erfc(x/(2*np.sqrt(1.3e-4*3600*t)))*(t*(48-t))

def new_solve_dif(T, dt, N, D):

    # Define the domain in mm
    mesh = IntervalMesh(N, 0, 10)

    # Define time discretization parameter
    num_steps = int(T/dt)

    # Define the finite element spaces and functions
    # element = FiniteElement('CG', mesh.ufl_cell(), 1)

    V = FunctionSpace(mesh, "CG", 1)
    C = TrialFunction(V)
    v = TestFunction(V)

    # Define Dirichlet boundary condition depending on time
    # time to be modified time based on the time discretization
    C_D = Expression('t*(48-t)', t = 0, degree = 1)

    # Define the left boundary domain
    def GammaL(x, on_boundary):
        return near(x[0], 0)
    
    # Define the right boundary domain
    def GammaR(x, on_boundary):
        return near(x[0], 10)
    
    b_fun = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    b_fun.array()[0] = 1

    AutoSubDomain(GammaR).mark(b_fun, 1)

    # ds = Measure('ds', domain=mesh)
    
    # Define the Dirichlet condition
    bc = DirichletBC(V, C_D, GammaL)

    # Define the right-hand size
    g = Expression('(48 - 2*t)*erfc(x[0]/(2*sqrt(D*t)))', t = 1e-16, D = D, degree = 1)

    # Define the Neumann condition
    h = Expression('-(sqrt(D)*t*(48-t))/(sqrt(pi*t))*exp(-L*L/(4*D*t))', t = 1e-16, L=10, D = D, degree = 1)

    # Define the analytic solution
    C_e = Expression('t*(48-t)*erfc(x[0]/(2*sqrt(D*t)))', t = 1e-16, D = D, degree = 1)

    # Define initial value
    C_n = Function(V)

    
    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - dt*g*v*dx - C_n*v*dx + dt*h*v*ds(1, subdomain_data = b_fun)
    a, L = lhs(F), rhs(F)

    # Solution at final time T
    C_T = Function(V)

    t = 0
    for n in tqdm(range(num_steps)):
        
        # Update current time
        t += dt

        # Update current Dirichlet conditon
        C_D.t = t

        # Update current Neumann condition
        h.t = t

        # Update current right-hand side
        g.t = t

        # Update current exact solution
        C_e.t = t

        # Compute solution solving the variational problem
        solve(a == L, C_T, bc)

        # Assign the initial intial value for next time step
        C_n.assign(C_T)

    # This function retutns the solution a final time T
    return C_T, C_e


# for k in range(3):
#     # Final time = 1h, time step = 0.01h, D = 1.3e-4*3600 mm^2/h 
#     N = 10*(2**k)                    
#     print ("solving for ", N)
#     C_T = new_solve_dif(1, 0.01, N, 1.3e-4*3600)[0]
#     V = C_T.function_space()
#     plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), linewidth=2)
# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Concentration [mm$^{{-3}}$]')
# legend = [f"N={10*(2**k)}" for k in range(3)]
# plt.legend(legend, loc = 1)
# plt.title('Solution of Diffusion problem with right-hand side after 1 hour')
# plt.show()


# N_values = []
# error_L2_values = []
# error_H1_values = []
# error_Linf_values = []
# h = []


# for k in tqdm(range(7)):

#     N = 10*(2**k)
#     h.append(10/N)
#     print("Solving for N =", N)
    
#     # Risolvi il problema per il valore corrente di N
#     C_T, C_e = new_solve_dif(1, 0.0001, N, 1.3e-4*3600)
    
#     error_L2 = errornorm(C_e, C_T, 'L2')
#     error_H1 = errornorm(C_e, C_T, 'H1')

#     num_points = N  # Numero di punti sulla griglia
#     x_values = np.linspace(0, 10, num_points)  # Punti lungo l'asse x
#     C_e_values = np.array([C_e(x) for x in x_values])
#     C_T_values = np.array([C_T(x) for x in x_values])
#     error_Linf = np.max(np.abs(C_e_values - C_T_values))
    
#     N_values.append(N)
#     error_L2_values.append(error_L2)
#     error_H1_values.append(error_H1)
#     error_Linf_values.append(error_Linf)

# print('errors L2:', error_L2_values)
# print('errors H1:', error_H1_values)
# print('errors Linf:', error_Linf_values)
# print(N_values)

# p_L2 = []
# p_H1 = []
# p_Linf = []

# for i in range(1, len(error_L2_values)):
#     p_L2_errors = np.log(error_L2_values[i]/error_L2_values[i-1])/np.log(h[i]/h[i-1])
#     p_L2.append(p_L2_errors)
#     p_H1_errors = np.log(error_H1_values[i]/error_H1_values[i-1])/np.log((h[i]/h[i-1]))
#     p_H1.append(p_H1_errors)
#     p_Linf_errors = np.log(error_Linf_values[i]/error_Linf_values[i-1])/np.log((h[i]/h[i-1]))
#     p_Linf.append(p_Linf_errors)

# print("Slope L2:", p_L2)
# print("Slope H1:", p_H1)
# print("Slope Linf:", p_Linf)

# plt.figure()
# plt.loglog(N_values, error_L2_values, marker='o', label='L2 Error')
# plt.xlabel('N')
# plt.ylabel('Error (log scale)')
# plt.legend()
# plt.title('L2 Error vs. N (log scale)')

# plt.figure()
# plt.loglog(N_values, error_H1_values, marker='o', label='H1 Error')
# plt.xlabel('N')
# plt.ylabel('Error (log scale)')
# plt.legend()
# plt.title('H1 Error vs. N (log scale)')

# plt.figure()
# plt.loglog(N_values, error_Linf_values, marker='o', label='Linf Error')
# plt.xlabel('N')
# plt.ylabel('Error (log scale)')
# plt.legend()
# plt.title('Linf Error vs. N (log scale)')

# plt.show()

# # Calcola il logaritmo dei valori di N e degli errori L2
# log_N_values = np.log(N_values)
# log_error_L2_values = np.log(error_L2_values)

# # Esegui la regressione lineare per calcolare il coefficiente angolare della retta in scala logaritmica
# slope_L2, intercept_L2, r_value_L2, p_value_L2, std_err_L2 = linregress(log_N_values, log_error_L2_values)

# # Calcola il logaritmo dei valori di N e degli errori H1
# log_error_H1_values = np.log(error_H1_values)

# # Esegui la regressione lineare per calcolare il coefficiente angolare della retta in scala logaritmica
# slope_H1, intercept_H1, r_value_H1, p_value_H1, std_err_H1 = linregress(log_N_values, log_error_H1_values)

# log_error_Linf_values = np.log(error_Linf_values)

# slope_Linf, intercept_Linf, r_value_Linf, p_value_Linf, std_err_Linf = linregress(log_N_values, log_error_Linf_values)

# # Stampa i coefficienti angolari delle rette
# print("Slope L2:", slope_L2)
# print("Slope H1:", slope_H1)
# print("Slope Linf:", slope_Linf)


# C_T= new_solve_dif(8, 1, 100, 1.3e-4*3600)[0]
# V = C_T.function_space()
# space = np.arange(0, 10 + 0.01, 0.1)
# plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), color ='red', label = 'Numerical Solution')
# plt.plot(space, anal_sol(space, 8), color = 'blue', linestyle='--', label = 'Analytic Solution')
# plt.xlabel(r'distance [mm]')
# plt.ylabel(r'concentration [mm$^{{-3}}$]')
# plt.legend()
# plt.title('Comparison between Numerical and Analytical solution after 8 hours, $\Delta t = 1$ hour')
# plt.show()

# space = np.linspace(0, 10, 100)  
# time = np.linspace(0, 48, int(48/1))  

# # Plot 2
# # Initialize an empty list to store the curves
# concentration_curves = []

# # Calculate concentration curves for various time points
# for t in time:
#     C_T = new_solve_dif(t, 1, 100, 1.3e-4*3600)[0]
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

# # Calculate concentration values for different points in time and space
# for i, t in enumerate(time):
#     C_T = new_solve_dif(t, 1, 100, 1.3e-4*3600)[0]
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

# Comparison between diffusion with and without rhs
# C_T = solve_dif(8, 0.1, 100, 1.3e-4*3600)
# C_T_rhs = new_solve_dif(8, 0.1, 100, 1.3e-4*3600)[0]
# V = C_T.function_space()
# plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), linewidth=2, color = 'red', label ='Diffusion without rhs')
# plt.plot(V.tabulate_dof_coordinates(), C_T_rhs.vector().get_local(), linewidth=2, color = 'blue', label ='Diffusion with rhs')
# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Concentration [mm$^{{-3}}$]')
# plt.legend()
# plt.title(r'Comparison between diffusion solution with and wthout right-hand side after 1 hour, $\Delta$t = 6 minutes')
# plt.show()

# Plot the analytical solution
# Define the values of t for which you want to plot the function
# t = 24.0

# Create an array of x-values
# x_values = np.arange(0, 10, 0.01)  # Adjust the range and number of points as needed

# Plot the function for each value of t

# C_values = erfc(x_values / (2 * np.sqrt(1.3e-4 * 3600 * t)))
# C_values_2 = erfc(x_values / (2 * np.sqrt(1.3e-4 * 3600 * t))) * (t * (48 - t))
# C_values = anal_sol(x_values, 24)
# C_values_2 = anal_sol_2(x_values, 24)
# plt.plot(x_values, C_values, label=r'Analytical solution $\tilde{C}$')
# plt.plot(x_values, C_values_2, label=r'Analytical solution $\hat{C}$')




