"""
We consider the following problem

du/dt - div(D*grad(u)) = f(t) in Omega

Omega = (0, L)    L = 1 cm
D = 1.2*10**-4 mm^2/s

We want to know u after 8, 24 and 48 hours

Initial condition
u(x,0) = 0 in Omega

Buondary conditions
- Dirichlet buondary condition
u(0,t) = 0

- Neumann boundary condition
-D*grad(u)*n = 0 in x = L 

- Source Term
f(t) = t*(48-t)
"""

from __future__ import print_function
from fenics import *
from dolfin_adjoint import *
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

def solve_dif_source(T, dt, N, D):

    mesh = IntervalMesh(N, 0, 10)

    num_steps = int(T/dt)

    # Define the finite element spaces and functions
    V = FunctionSpace(mesh, "CG", 1)
    C = TrialFunction(V)
    v = TestFunction(V)         

    # Define source term depending on time
    # f = Expression('t*(48 - t)', t=0, degree=1)  

    # Define Dirichlet boundary condition

    C_D = Constant(0)

    def GammaL(x, on_boundary):
            return near(x[0], 0)

    bc = DirichletBC(V, C_D, GammaL) 

    # Define source term

    f = Expression('t*(48 - t)', t=0, degree=2)

    # Define initial value
    C_n = Function(V)

    # Define the variational problem to be solved at each time
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx - f*v*dx
    a, L = lhs(F), rhs(F)

    # Time-stepping
    C_T = Function(V)

    t = 0
    for n in tqdm(range(num_steps)):
        # Update current time
        t += dt

        # Update current source term
        f.t = t 
                
        # Compute solution
        solve(a == L, C_T, bc)

        C_n.assign(C_T)

    return C_T

for N in [10, 200, 4000]:
    # Final time = 1h, time step = 6mins, D = 1.3e-4*3600 mm^2/h                     
    print ("solving for ", N)
    C_T = solve_dif_source(1, 0.1, N, 1.3e-4*3600)
    V = C_T.function_space()
    plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), linewidth=2)
    plt.xlabel(r'distance [mm]')
    plt.ylabel(r'concentration [mm$^{{-3}}$]')
    legend = ["N=%d"%N for N in [10, 200, 4000]]
    plt.legend(legend, loc = 4)
    plt.title('Final time = 1h, time step = 6mins, D = 1.3e-4*3600 mm^2/h, source term')

plt.savefig('1hour_6mins_source term')
plt.show()



