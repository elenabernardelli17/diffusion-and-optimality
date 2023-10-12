
from __future__ import print_function
from dolfin import *
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.special import erfc
import numpy as np

def anal_sol(x,t):
     
    return erfc(x/(2*np.sqrt(1.3e-4*3600*t)))*(t*(48-t))


mesh = IntervalMesh(100, 0, 10)
P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
R = FiniteElement('R', mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([P1, R]))
C, lam = TrialFunctions(W)
v, q = TestFunctions(W)
sol_n = Function(W)

C_D = Expression('t*(48 - t)', t=0, degree=1)

D = Constant(1.3e-4*3600)

T = 48 # hours
dt = 0.01 # timestep of 1 hours
num_steps = int(T/dt)

n = FacetNormal(mesh)

b_fun = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
b_fun.array()[0] = 1

F = C*v*dx - sol_n[0]*v*dx + dt*D*dot(grad(C), grad(v))*dx - dt*C_D*v*ds(1, subdomain_data = b_fun) + q*dt*(C - C_D)*ds(1, subdomain_data = b_fun) + lam*dt*v*ds(1, subdomain_data = b_fun)
a, L = lhs(F), rhs(F)

sol = Function(W)

t = 0
for n in range(1, num_steps):
    # Update current time
    t += dt
    C_D.t = t 
            
    # Compute solution
    solve(a == L, sol)
    sol_n.assign(sol)

C_T = sol.split()[0]

lamb = C_T.vector().get_local()[-1]

V = FunctionSpace(mesh, 'CG', 1)

space = np.arange(0, 10 + 0.1, 0.1)

plt.figure()
plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local()[:-1], label = 'numerical solution')
plt.plot(space, anal_sol(space, 24), label = 'analytical solution')
plt.xlabel(r'distance [mm]')
plt.ylabel(r'concentration [mm$^{{-3}}$]')
plt.legend()
plt.title('Lagrange multiplier = 515.45, T = 24 hours, $\Delta t = 36$ seconds')
print(lamb)
# plt.savefig('Lagrange_multiplier')
plt.show()




    



