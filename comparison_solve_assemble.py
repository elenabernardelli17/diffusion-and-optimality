from __future__ import print_function
from fenics import *
from dolfin_adjoint import *
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt


from scipy.special import erfc


numerical_solutions_solve = []
numerical_solutions_assemble = []

# Define the domain
mesh = IntervalMesh(100, 0, 10)

# Define time discretization parameters
T = 1 # hours   
dt = 0.1 # timestep of 6 mins
num_steps = int(T/dt)

# Define the diffusion parameter
D = Constant(1.3e-4*3600) #mm^2/h

# Define the finite element spaces and functions
V = FunctionSpace(mesh, "CG", 1)
C = TrialFunction(V)
v = TestFunction(V)         

# Define Dirichlet buondary condition depending on time 
C_D = Expression('t*(48 - t)', t=0, degree=1)  

# Define the left boundary domain
def GammaL(x, on_boundary):
        return near(x[0], 0)

bc = DirichletBC(V, C_D, GammaL) 

# Define initial value
C_n = Function(V)

# Define the variational problem to be solved at each time
F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
a, L = lhs(F), rhs(F)

t = 0
C_solve = Function(V, name = 'C_solve')
for n in tqdm(range(num_steps)):
    
    # Update current time
    t += dt
    # Update Dirichlet condition
    C_D.t = t 
            
    # Compute solution
    solve(a == L, C_solve, bc, solver_parameters = {'linear_solver':'lu'})

    C_n.assign(C_solve)
    
numerical_solutions_solve.append((V.tabulate_dof_coordinates(), C_solve.vector()))

C_n.assign(Constant(0.0))

t = 0
C_assemble = Function(V, name = 'C_assemble')
for n in tqdm(range(num_steps)):
   
    # Update current time
    t += dt
    # Update Dirichlet condition
    C_D.t = t

    b = assemble(L)
    A = assemble(a)
    bc.apply(A, b)
    solve(A, C_assemble.vector(), b, 'lu')
    C_n.assign(C_assemble)
numerical_solutions_assemble.append((V.tabulate_dof_coordinates(), C_assemble.vector()))

# Print solve and assemble solutions 
x = V.tabulate_dof_coordinates()

plt.plot(x, C_solve.vector().get_local(), label="solve")
plt.plot(x, C_assemble.vector().get_local(), label="assemble")
plt.legend()
plt.title('Comparison solve assemble solutions')
plt.show()
plt.savefig('Comparison solve assemble solutions')


