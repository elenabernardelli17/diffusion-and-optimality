from __future__ import print_function
from fenics import *
from dolfin_adjoint import *
import numpy as np

import ipyopt
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.interpolate import interp1d

# Function to solve my diffusion problem 
def solve_dif(T, dt, N, D_min, D_max):
    
    # Define the domain in mm
    mesh = IntervalMesh(N, 0, 10)

    # Define time discretization parameter
    num_steps = int(T/dt)

    # Define the finite element spaces and functions
    V = FunctionSpace(mesh, "CG", 1)
    C = TrialFunction(V)
    v = TestFunction(V)        

    # Define the subdomains where you want to assign the minimum coefficient
    # class MinCoefficientsRegion1(SubDomain):
    #     def inside(self, x, on_boundary):
    #         # Define the subdomain where D_min should be applied (region 3)
    #         return 0.0 < x[0] < 1.0
    
    class MinCoefficientsRegion2(SubDomain):
        def inside(self, x, on_boundary):
            # Define the subdomain where D_min should be applied (region 1)
            return 2.0 < x[0] < 4.0

    class MinCoefficientsRegion3(SubDomain):
        def inside(self, x, on_boundary):
            # Define the subdomain where D_min should be applied (region 2)
            return 6.0 < x[0] < 8.0

    # Create instances of the subdomain markers
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    subdomains.set_all(0)

    # Mark the subdomains with different values
    # MinCoefficientsRegion1().mark(subdomains, 1)
    MinCoefficientsRegion2().mark(subdomains, 1)
    MinCoefficientsRegion3().mark(subdomains, 1)
        
    # Initialize the coefficient function
    D = Function(V)
    D.interpolate(Constant(D_max))

    # Assign values to the coefficient function based on subdomain markers
    D_values = D.vector()
    D_array = D_values.get_local()
    for cell in cells(mesh):
        if subdomains[cell] == 0:
            D_array[cell.index()] = D_max
        else:
            D_array[cell.index()] = D_min

    D_values.set_local(D_array)
    D_values.apply('insert')

    # Define Dirichlet buondary condition depending on time
    # time to be modified time based on the time discretization
    C_D = Expression('t*(48 - t)', t=0, degree=2)        

    # Define the left boundary domain
    def GammaL(x, on_boundary):
            return near(x[0], 0)

    # Define the Dirichlet condition 
    bc = DirichletBC(V, C_D, GammaL) 

    # Define initial value
    C_n = Function(V) 

    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*inner(grad(C), grad(v))*dx - C_n*v*dx
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
    return C_T, D


C_T, D = solve_dif(1, 0.1, 100, 0.6e-4*3600, 1.3e-4*3600)
V = C_T.function_space()
plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), linewidth=2)
plt.xlabel('Distance [mm]')
plt.ylabel('Concentration [mm$^{-3}$]')
plt.title('Diffusion with Piecewise linear function after 48 hours')

# # Compute piecewise function for D on the mesh 
# mesh = IntervalMesh(100, 0, 10)
# D_values = D.compute_vertex_values(mesh)

# # Plot piecewise function for the diffusion coefficient 
# plt.figure()
# plt.plot(V.tabulate_dof_coordinates(), D_values, label = 'Diffusion Coefficient')
# plt.xlabel("Distance [mm]")
# plt.ylabel("D [mm$^{-3}$/h]")
# plt.legend()
# plt.title('Piecewise function describing the Diffusion Coefficient')

plt.show()

# space = np.linspace(0, 10, 100)  
# time = np.linspace(0, 48, int(48/1))  

# # Plot 2
# # Initialize an empty list to store the curves
# concentration_curves = []

# Calculate concentration curves for various time points
# for t in time:
#     C_T = solve_dif(t, 1, 100, 0.6e-4*3600, 1.3e-4*3600)[0]
#     concentration_values = [C_T(x) for x in space]
#     concentration_curves.append(concentration_values)

# Plot concentration curves with different colors
# for i, curve in enumerate(concentration_curves):
#     plt.plot(space, curve, label=f'Time: {time[i]:.1f}')

# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Concentration [mm$^{{-3}}$]')
# # plt.legend()
# plt.title('Concentration Evolution Over Time')
# plt.show()


# Initialize a matrix to store concentration values
# concentration_matrix = np.zeros((len(time), len(space)))

# Calculate concentration values for different points in time and space
# for i, t in enumerate(time):
#     C_T = solve_dif(t, 1, 100, 0.6e-4*3600, 1.3e-4*3600)[0]
#     for j, x in enumerate(space):
#         concentration_matrix[i, j] = C_T(x)

# Create a heatmap-like plot to represent diffusion
# plt.imshow(concentration_matrix, extent=[space.min(), space.max(), time.min(), time.max()],
#            aspect='auto', cmap='viridis', origin='lower')

# Add vertical dividing lines
# num_vertical_lines = 5  # Adjust the number of vertical lines as needed
# space_dividers = np.linspace(space.min(), space.max(), num_vertical_lines + 1)
# for space_divider in space_dividers[1:-1]:
#     plt.axvline(x=space_divider, color='white', linestyle='--', linewidth=1)
# plt.colorbar(label=r'Concentration [mm$^{{-3}}$]')
# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Time [h]')
# plt.title('Diffusion in Time and Space')
# plt.show()

# Create a filled contour plot to represent diffusion
# plt.contourf(space, time, concentration_matrix, cmap='viridis', levels=20)
# plt.colorbar(label=r'Concentration [mm$^{{-3}}$]')
# plt.xlabel(r'Distance [mm]')
# plt.ylabel(r'Time [h]')
# plt.title('Diffusion in Time and Space')
# plt.show()
