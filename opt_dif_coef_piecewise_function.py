from fenics import *
# from fenics_adjoint import *
from pyadjoint import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import erfc


def solve_dif_data(V, T, dt):
    num_steps = int(T/dt)

    mesh = IntervalMesh(100, 0, 10)
    
    C = TrialFunction(V)
    v = TestFunction(V)

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
        

    # Function space for D
    V_D = FunctionSpace(mesh, "DG", 1)

    D_max = Constant(1.3e-4*3600)
    D_min = Constant(0.6e-4*3600)
    # Initialize the coefficient function
    D = interpolate(Constant(D_max), V_D)

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

    D_new = interpolate(Constant(0.1), V_D)

    # Define the integral terms with associated domains
    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)

    # Define Dirichlet buondary condition depending on time
    # time to be modified time based on the time discretization
    C_D = Expression('t*(48 - t)', t=0, degree = 1) 

    # Define the left boundary domain
    def GammaL(x, on_boundary):
            return near(x[0], 0)

    # Define the Dirichlet condition 
    bc = DirichletBC(V, C_D, GammaL) 

    # Define the initial value
    C_n_ref = Function(V)

    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n_ref*v*dx
    a, L = lhs(F), rhs(F)

     # Solution at final time T
    C_T_ref = Function(V)

    t = 0

    J_values = []

    for n in range(num_steps):
        # Update time
        t += dt
        # Update Dirichlet boundary condition
        C_D.t = t 
        
        # Solve PDE
        solve(a == L, C_T_ref, bc)

        C_n_ref.assign(C_T_ref)

    return C_T_ref

from fenics_adjoint import *

def optimal_dif_function(T, dt, N):

    mesh = IntervalMesh(N, 0, 10)

    num_steps = int(T/dt)

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
        

    # Function space for D
    V_D = FunctionSpace(mesh, "DG", 1)

    D_max = Constant(0.2e-4*3600)
    D_min = Constant(0.1e-4*3600)

    # Initialize the coefficient function
    D = interpolate(Constant(D_max), V_D)

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

    # Define the integral terms with associated domains
    dx = Measure("dx", domain=mesh, subdomain_data=subdomains)

    # Define Dirichlet buondary condition depending on time
    # time to be modified time based on the time discretization
    C_D = Expression('t*(48 - t)', t=0, degree = 1) 

    # Define the left boundary domain
    def GammaL(x, on_boundary):
            return near(x[0], 0)

    # Define the Dirichlet condition 
    bc = DirichletBC(V, C_D, GammaL) 

    # Define the initial value
    C_n = Function(V)

    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
    a, L = lhs(F), rhs(F)

     # Solution at final time T
    C_T = Function(V)

    J = 0
    t = 0

    J_values = []

    C_T_ref = solve_dif_data(V, 1, 0.1)

    for n in range(num_steps):
        # Update time
        t += dt
        # Update Dirichlet boundary condition
        C_D.t = t 
        
        # Solve PDE
        solve(a == L, C_T, bc)

        C_n.assign(C_T)

        # Cost functional 
        J += 0.5*float(dt)*assemble(inner(C_T - C_T_ref, C_T - C_T_ref)*dx) + alpha/2*float(dt)*assemble(inner(D, D)*dx)
        print("Step:", n, "t:", t, "J:", J)
        J_values.append(J)

    control = Control(D)

    minimized_functional = ReducedFunctional(J, control)

    # Minimization of the cost function   
    D_optimal = minimize(minimized_functional, method = 'BFGS', tol = 1e-10, options = {'disp':True})

    # print('The optimal diffusion coefficient is:', optimal_diffusion)

    return D_optimal, D

alpha_values = [1, 10, 100, 1000]
space = np.linspace(0, 10, 100)

for alpha in alpha_values:
    D_optimal = optimal_dif_function(1, 0.1, 100)[0]

    # Extract the values of the optimal solution at mesh points
    solution_values = [D_optimal(x) for x in space]

    plt.plot(space, solution_values, label=f'Alpha = {alpha}')

plt.legend()
plt.xlabel(r'Distance [mm]')
plt.ylabel('Optimal Solution')
plt.title('Optimal Diffusion Piecewise Linear Function')
plt.show()

# D = optimal_dif_coef(1, 0.1, 100, Constant(0.6e-4*3600), Constant(1.3e-4*3600), 1)[1]
# mesh = IntervalMesh(100, 0, 10)
# D_values = D.compute_vertex_values(mesh)

# plt.figure()
# C_T = optimal_dif_coef(1, 0.1, 100, Constant(0.6e-4*3600), Constant(1.3e-4*3600), 1)[2]
# V = C_T.function_space()
# plt.plot(V.tabulate_dof_coordinates(), D_values, label = 'Diffusion Coefficient')
# plt.xlabel("Distance [mm]")
# plt.ylabel("D [mm$^{-3}$/h]")
# plt.legend()
# plt.title('Piecewise function describing the Diffusion Coefficient')

# plt.show()


# print_optimization_methods()
# L-BFGS-B :  The L-BFGS-B implementation in scipy.
# SLSQP :  The SLSQP implementation in scipy.
# TNC :  The truncated Newton algorithm implemented in scipy.
# CG :  The nonlinear conjugate gradient algorithm implemented in scipy.
# BFGS :  The BFGS implementation in scipy.
# Nelder-Mead :  Gradient-free Simplex algorithm.
# Powell :  Gradient-free Powells method
# Newton-CG :  Newton-CG method
# Anneal :  Gradient-free simulated annealing
# basinhopping :  Global basin hopping method
# COBYLA :  Gradient-free constrained optimization by linear approxition method
# Custom :  User-provided optimization algorithm