
from __future__ import print_function
from dolfin import *
import tqdm as tqdm
import matplotlib.pyplot as plt
from solve_diffusion import solve_dif

# Lagrange
mesh = IntervalMesh(100, 0, 10)
P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
R = FiniteElement('R', mesh.ufl_cell(), 0)
# V = FunctionSpace(mesh, "CG", 1)
# R = FunctionSpace(mesh, 'R', 0)
W = FunctionSpace(mesh, MixedElement([P1, R]))
C, lam = TrialFunctions(W)
v, q = TestFunctions(W)
sol_n = Function(W)


C_D = Expression('t*(48 - t)', t=1e-16, degree=1)

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

C_T_Lagrange = sol.split()[0]

# Nitsche
mesh = IntervalMesh(100, 0, 10)

V = FunctionSpace(mesh, "CG", 1)
C = TrialFunction(V)
v = TestFunction(V)
C_n = Function(V)

C_D = Expression('t*(48 - t)', t=0, degree=1)

D = Constant(1.3e-4*3600)

T = 48 # hours
dt = 0.01 # timestep of 1 hours
num_steps = int(T/dt)

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# lam = C_T_Lagrange.vector().get_local()[-1]
lam = 1000


b_fun = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
b_fun.array()[0] = 1

F = C*v*dx - C_n*v*dx + D*dt*inner(grad(C), grad(v))*dx - D*dt*inner(grad(C), n)*v*ds(1, subdomain_data = b_fun) - D*dt*C*inner(grad(v),n)*ds(1, subdomain_data = b_fun) + D*dt*C_D*inner(grad(v),n)*ds(1, subdomain_data = b_fun) + (lam/h)*dt*C*v*ds(1, subdomain_data = b_fun) - (lam/h)*dt*C_D*v*ds(1, subdomain_data = b_fun)
a, L = lhs(F), rhs(F)

C_T_Nitsche = Function(V)
C_n.assign(C_T_Nitsche)

t = 0
for n in range(1, num_steps):
    # Update current time
    t += dt
    C_D.t = t 
        
    # Compute solution
    C_n.assign(C_T_Nitsche)
    solve(a == L, C_T_Nitsche)


V = FunctionSpace(mesh, 'CG', 1)

plt.figure()
plt.plot(V.tabulate_dof_coordinates(), C_T_Lagrange.vector().get_local()[:-1], color = 'red', label = 'Lagrange multiplier method')
plt.plot(V.tabulate_dof_coordinates(), C_T_Nitsche.vector().get_local(), color = 'blue', linestyle='--', label = 'Nitsche method')
plt.legend()
plt.xlabel(r'distance [mm]')
plt.ylabel(r'concentration [mm$^{{-3}}$]')
plt.title('Comparison between Nitsche method and Lagrange multiplier method')

print(lam)
plt.show()

C_T = solve_dif(48, 0.01, 100, 1.3e-4*3600)

plt.figure()
plt.plot(V.tabulate_dof_coordinates(), C_T_Lagrange.vector().get_local()[:-1], color = 'red', label = 'Lagrange multiplier method')
plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), color = 'blue', linestyle='--', label = 'Solution with classical approach')
plt.legend()
plt.xlabel(r'distance [mm]')
plt.ylabel(r'concentration [mm$^{{-3}}$]')
plt.title('Comparison between Lagrange multiplier method and solution with classical approach')
plt.show()


