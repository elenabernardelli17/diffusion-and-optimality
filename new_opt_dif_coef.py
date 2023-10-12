from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



# def solve_dif(V, T, dt, D):
    

#     # Define time discretization parameter
#     num_steps = int(T/dt)

#     # Define the finite element spaces and functions
#     C = TrialFunction(V)
#     v = TestFunction(V)         

#     # Define Dirichlet buondary condition depending on time
#     # time to be modified time based on the time discretization
#     C_D = Expression('t*(48 - t)', t=0, degree=1)        

#     # Define the left boundary domain
#     def GammaL(x, on_boundary):
#             return near(x[0], 0)

#     # Define the Dirichlet condition 
#     bc = DirichletBC(V, C_D, GammaL) 

#     # Define initial value
#     C_n = Function(V) 

#     # Define the variational problem to be solved at each time step
#     F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
#     a, L = lhs(F), rhs(F)

#     # Solution at final time T
#     C_T = Function(V)

#     solution_functions = []
#     t = 0
#     for n in tqdm(range(num_steps)):

#         # Update current time
#         t += dt

#         # Update current Dirichlet condition
#         C_D.t = t 
                
#         # Compute solution solving the variational problem
#         solve(a == L, C_T, bc)

#         # Append the solution function to the list
#         solution_functions.append(C_T)

#         # Assign the ne initial value for next time step
#         C_n.assign(C_T)
    
#     # This function returns the solution at final time T
#     return C_T, solution_functions

# # Define the parameters
# T = 1 # hour
# N = 100
# dt = 0.1

# num_steps = int(T/dt)

# # Create the mesh, function space, and functions
# mesh = IntervalMesh(N, 0, 10)
# V = FunctionSpace(mesh, "CG", 1)
# C = TrialFunction(V)
# v = TestFunction(V)

# # Define Dirichlet buondary condition depending on time
# # time to be modified time based on the time discretization
# C_D = Expression('t*(48 - t)', t=0, degree=1) 

# # Define the left boundary domain
# def GammaL(x, on_boundary):
#     return near(x[0], 0)

# # Define the Dirichlet condition 
# bc = DirichletBC(V, C_D, GammaL) 

# # Define initial value
# C_n = Function(V)

# # Define the variational problem to be solved at each time step
# D = Constant(0.8)
# F = C * v * dx + dt * D * dot(grad(C), grad(v)) * dx - C_n * v * dx
# a, L = lhs(F), rhs(F)

# # Solution at final time T
# C_T = Function(V)

# t = 0
# J = 0

# # Solve the diffusion problem and get the solution functions from solve_dif
# C_T, solution_functions = solve_dif(V, T, dt, 1.3e-4*3600)

# for n in range(num_steps):
#     t += dt

#     # Update Dirichlet condition
#     C_D.t = t
#     solve(a == L, C_T, bc)

#     C_n.assign(C_T)

#     data = solution_functions[n]

#     J += 0.5 * dt * assemble(inner(C_T - data, C_T - data) * dx)

# control = Control(D)

# minimized_functional = ReducedFunctional(J, control)

# D_optimal = minimize(minimized_functional, method='BFGS', tol=1e-3, options = {'disp':True})

# optimal_diffusion = float(D_optimal)

# print('The optimal diffusion coefficient is:', optimal_diffusion)

def solve_dif_data(V, T, dt):
    
    # Define the domain in mm
    # mesh = IntervalMesh(100, 0, 10)

    # Define time discretization parameter
    num_steps = int(T/dt)

    # Define the finite element spaces and functions
    # V = FunctionSpace(mesh, "CG", 1)
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
    C_n_ref = Function(V) 

    D = Constant(1.3e-4*3600)

    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n_ref*v*dx
    a, L = lhs(F), rhs(F)

    # Solution at final time T
    C_T_ref = Function(V)

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
        solve(a == L, C_T_ref, bc)
        
        # Assign the ne initial value for next time step
        C_n_ref.assign(C_T_ref)
    
    # This function returns the solution at final time T
    return C_T_ref


# C_T_ref = Function(V)
# C_T_ref = C_T_new.vector().get_local().copy()
# C_T_ref[:] += 1e-4*np.random.randn(C_T_ref.shape[0])

# C_T_new.vector()[:] = C_T_new_vec

from fenics_adjoint import *


def solve_dif(T, dt, N):
    
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

    # D = Constant(1.3e-4*3600)
    D = Constant(0.2e-4*3600)

    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
    a, L = lhs(F), rhs(F)

    # Solution at final time T
    C_T = Function(V)

    solution_functions = []
    t = 0
    J = 0

    C_T_ref = solve_dif_data(V, 1, 0.1)
    # data = C_T_ref.vector().get_local().copy()
    # print(type(data))
    for n in tqdm(range(num_steps)):

        # Update current time
        t += dt

        # Update current Dirichlet condition
        C_D.t = t 
                
        # Compute solution solving the variational problem
        solve(a == L, C_T, bc)

        # # Append the solution function to the list
        # solution_functions.append(C_T)

        # Assign the ne initial value for next time step
        C_n.assign(C_T)

        # Update cost functional
        J += 0.5*dt*assemble(inner(C_T - C_T_ref, C_T - C_T_ref)*dx)

    control = Control(D)

    minimized_functional = ReducedFunctional(J, control)

    D_optimal = minimize(minimized_functional, method='BFGS', tol=1e-10)

    optimal_diffusion = float(D_optimal)

    print('The optimal diffusion coefficient is:', optimal_diffusion)

solve_dif(1, 0.01, 100)



