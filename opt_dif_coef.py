from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import erfc

def optimal_dif_coef(T, dt, N, D):

    mesh = IntervalMesh(N, 0, 10)

    num_steps = int(T/dt)

    V = FunctionSpace(mesh, "CG", 1)

    C_D = Expression('t*(48 - t)', t=0, degree = 1) 

    data = Expression("erfc(x[0]/(2*sqrt(D*t)))*(t*(48-t))", D = 1.3e-4*3600, t = 1e-16, degree=1)

    C = TrialFunction(V)
    v = TestFunction(V)

    # Define initial value
    C_T = Function(V)
    # d = Function(V, name="data")
    C_n = Function(V)

    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
    a, L = lhs(F), rhs(F)

    def GammaL(x, on_boundary):
        return near(x[0], 0)

    bc = DirichletBC(V, C_D, GammaL)

    J = 0.5*float(dt)*assemble(inner(C_T - data, C_T - data)*dx)

    t = float(0)

    J_values = []
    
    for n in tqdm(range(num_steps)):
        # Update time
        t += dt
        # Update Dirichlet boundary condition
        C_D.t = t 

        # Update data function
        data.t = t
        
        # Solve PDE
        solve(a == L, C_T, bc)

        C_n.assign(C_T)

        # Cost functional 
        J += 0.5*dt*assemble(inner(C_T - data, C_T - data)*dx) 
        print("Step:", n, "t:", t, "J:", J)
        J_values.append(J)

    control = Control(D)

    minimized_functional = ReducedFunctional(J, control)

    # Minimization of the cost function   
    D_optimal = minimize(minimized_functional, method = 'BFGS', tol = 1e-3, options = {'disp':True})

    optimal_diffusion = float(D_optimal)

    print('The optimal diffusion coefficient is:', optimal_diffusion)

    return optimal_diffusion

optimal_dif_coef(8, 0.001, 100, Constant(1.3e-4*3600))

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
















