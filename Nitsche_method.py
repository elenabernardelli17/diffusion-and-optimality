from __future__ import print_function
from dolfin import *
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.special import erfc
import numpy as np
from piecewise_diffusion_coefficient_ import solve_dif


def anal_sol(x,t):
     
    return erfc(x/(2*np.sqrt(1.3e-4*3600*t)))*(t*(48-t))



mesh = IntervalMesh(100, 0, 10)

V = FunctionSpace(mesh, "CG", 1)
C = TrialFunction(V)
v = TestFunction(V)
C_n = Function(V)

C_D = Expression('t*(48 - t)', t=1e-16, degree=1)

# D = Constant(1.3e-4*3600)
D_max = 1.3e-4*3600
D_min = 0.6e-4*3600

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


T = 24 # hours
dt = 0.1 # timestep 
num_steps = int(T/dt)
lam = 10000
h = CellDiameter(mesh)
n = FacetNormal(mesh)

b_fun = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
b_fun.array()[0] = 1

# F = C*v*dx - C_n*v*dx + D*dt*inner(grad(C), grad(v))*dx - D*dt*inner(grad(C), n)*v*ds(1, subdomain_data = b_fun) - D*dt*C*inner(grad(v),n)*ds(1, subdomain_data = b_fun) + D*dt*C_D*inner(grad(v),n)*ds(1, subdomain_data = b_fun) + (lam/h)*dt*C*v*ds(1, subdomain_data = b_fun) - (lam/h)*dt*C_D*v*ds(1, subdomain_data = b_fun)
# a, L = lhs(F), rhs(F)

F = C*v*dx - C_n*v*dx + dt*inner(D*grad(C), grad(v))*dx - dt*inner(D*grad(C), n)*v*ds(1, subdomain_data = b_fun) - dt*C*inner(D*grad(v),n)*ds(1, subdomain_data = b_fun) + dt*C_D*inner(D*grad(v),n)*ds(1, subdomain_data = b_fun) + (lam/h)*dt*C*v*ds(1, subdomain_data = b_fun) - (lam/h)*dt*C_D*v*ds(1, subdomain_data = b_fun)
a, L = lhs(F), rhs(F)

C_T = Function(V)
C_n.assign(C_T)

solution_list = []

t = 0
for n in range(1, num_steps):
    # Update current time
    t += dt
    C_D.t = t 
        
    # Compute solution
    C_n.assign(C_T)
    solve(a == L, C_T)

solution_list.append(C_T.vector().get_local())
space = np.arange(0, 10 + 0.1, 0.1)

# for i, solution in enumerate(solution_list):
#     plt.plot(V.tabulate_dof_coordinates(), solution, label= 'numerical solution')
#     plt.plot(space, anal_sol(space, 24), label = 'analytical solution')


# plt.xlabel(r'distance [mm]')
# plt.ylabel(r'concentration [mm$^{{-3}}$]')
# plt.title(r'Nitsche Method, $\lambda$ = 100, T = 24 hours, $\Delta t = 36$ seconds')
# plt.legend()
# plt.show()

C_T, D = solve_dif(24, 0.1, 100, 0.6e-4*3600, 1.3e-4*3600)

for i, solution in enumerate(solution_list):
    plt.plot(V.tabulate_dof_coordinates(), solution, label= 'Nitsche method', color = 'red')
    plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), label = 'Classical approach', color = 'blue', linestyle='--')


plt.xlabel(r'distance [mm]')
plt.ylabel(r'concentration [mm$^{{-3}}$]')
plt.title(r'Comparison between Nitsche method and classical approach with piecewise linear diffusion')
plt.legend()
plt.show()

