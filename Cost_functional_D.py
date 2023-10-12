from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt


def optimal_dif_coef(T, dt, D):

    mesh = IntervalMesh(100, 0, 10)

    num_steps = int(T/dt)

    V = FunctionSpace(mesh, "CG", 1)

    C_D = Expression('t*(48 - t)', t=0, degree = 2) 

    data = Expression("erfc(x[0]/(2*sqrt(D*t)))*(t*(48-t))", D = 1.3e-4*3600, t = 1e-16, degree=1)
    d = project(data, V)

    C = TrialFunction(V)
    v = TestFunction(V)

    # Define initial value
    C_T = Function(V)
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
        d.assign(project(data, V))
        
        # Solve PDE
        solve(a == L, C_T, bc)

        C_n.assign(C_T)

        # Cost functional 
        J += 0.5*dt*assemble(inner(C_T - data, C_T - data)*dx) 
        print("Step:", n, "t:", t, "J:", J)
        J_values.append(J)

    return J

def optimal_dif_coef_new(T, dt, D, D_new):

    mesh = IntervalMesh(100, 0, 10)

    num_steps = int(T/dt)

    V = FunctionSpace(mesh, "CG", 1)

    C_D = Expression('t*(48 - t)', t=0, degree = 2) 

    # data = Expression("erfc(x[0]/(2*sqrt(D*t)))*(t*(48-t))", D = 1.3e-4*3600, t = 1e-16, degree=1)
    # d = project(data, V)

    C = TrialFunction(V)
    C_new = TrialFunction(V)
    v = TestFunction(V)
    w = TestFunction(V)

    # Define initial value
    C_T = Function(V)
    C_n = Function(V)

    C_T_new = Function(V)
    C_n_new = Function(V)

    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
    a, L = lhs(F), rhs(F)

    F_new = C_new*w*dx + dt*D_new*dot(grad(C_new), grad(w))*dx - C_n_new*w*dx
    a_new, L_new = lhs(F_new), rhs(F_new)

    def GammaL(x, on_boundary):
        return near(x[0], 0)

    bc = DirichletBC(V, C_D, GammaL)

    # J = 0.5*float(dt)*assemble(inner(C_T - data, C_T - data)*dx)
    J = 0

    t = float(0)

    J_values = []
    
    for n in tqdm(range(num_steps)):
        # Update time
        t += dt
        # Update Dirichlet boundary condition
        C_D.t = t 

        # Update data function
        # data.t = t
        # d.assign(project(data, V))
        
        # Solve PDE
        solve(a == L, C_T, bc)

        solve(a_new == L_new, C_T_new, bc)

        C_n.assign(C_T)
        C_n_new.assign(C_T_new)

        # Cost functional 
        J += 0.5*dt*assemble(inner(C_T_new - C_T, C_T_new - C_T)*dx) 
        print("Step:", n, "t:", t, "J:", J)
        J_values.append(J)

    return J

D_values = np.arange(0.01, 1.5, 0.01)
J_values = []

for D in D_values:
    print('Solving for', D)
    J = optimal_dif_coef(1, 0.1, D)
    J_values.append(J)

min_J = min(J_values)
min_D_index = J_values.index(min_J)
optimal_D = D_values[min_D_index]

print("Optimla value for 'D':", optimal_D)

plt.plot(D_values, J_values, 'o-')
plt.xlabel('Diffusion Coefficient (D)')
plt.ylabel('Cost Functional (J)')
plt.title('Objective functional to determine the Optimal Diffusion Coefficient after 1 hour')
plt.grid(True)

plt.show()


