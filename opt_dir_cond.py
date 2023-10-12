from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import erfc

import ufl

def update_CD(params, t):
    C_D = ufl.max_value(params[0] * t * (params[1] - t), 0)
    return C_D

def opt_dir_cond(T, dt, N, D):

    mesh = IntervalMesh(N, 0 , 10)

    num_steps = int(T/dt)

    V = FunctionSpace(mesh, "CG", 1)
    C = TrialFunction(V)
    v = TestFunction(V)
    C_n = Function(V)

    # a = Constant(0.2)
    t = Constant(1e-16)
    # T_end = Constant(40)

    # C_D = Expression('a*t*(48 - t)', t=1e-16, a = a, degree=1)
    

    # params = Constant((1.2,1.0))
    params = [Constant(0.2), Constant(40)]
    # params[1]


    # C_D = ufl.max_value(t*(T_end - t),0)
    # C_D = ufl.max_value(params[0]*t*(params[1] - t),0)

    data = Expression("erfc(x[0]/(2*sqrt(D*t)))*(t*(48-t))", D = 1.3e-4*3600, t = 1e-16, degree=1)

    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    lam = 100

    b_fun = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    b_fun.array()[0] = 1

    F = C*v*dx - C_n*v*dx + D*dt*inner(grad(C), grad(v))*dx - D*dt*inner(grad(C), n)*v*ds(1, subdomain_data = b_fun) - D*dt*C*inner(grad(v),n)*ds(1, subdomain_data = b_fun) + D*dt*update_CD(params, t)*inner(grad(v),n)*ds(1, subdomain_data = b_fun) + (lam/h)*dt*C*v*ds(1, subdomain_data = b_fun) - (lam/h)*dt*update_CD(params, t)*v*ds(1, subdomain_data = b_fun)
    A, L = lhs(F), rhs(F)

    C_T = Function(V)
    C_n.assign(C_T)

    #t = 0
    J = 0
    alpha = 10
    for n in range(1, num_steps):
        # Update current time
        #t += dt
        #C_D.t = t
        t.assign(float(t) + dt) 
        data.t = float(t)

        C_D = update_CD(params, t)
            
        # Compute solution
        C_n.assign(C_T)
        solve(A == L, C_T)

        J += 0.5*dt*assemble(inner(C_T - data, C_T - data)*dx) #+ alpha/2*dt*assemble(inner(C_D, C_D)*ds(1, subdomain_data = b_fun))

    control = [Control(params[0]), Control(params[1])]

    # minimized_functional = ReducedFunctional(J, [a, T_end])
    minimized_functional = ReducedFunctional(J, control)

    # Minimization of the cost function   
    optimal = minimize(minimized_functional, method = 'BFGS', tol = 1e-10, options = {'disp':True})

    optimal_a = float(optimal[0])
    optimal_T_end = float(optimal[1])

    print('The optimal Dirichlet condition is with a =', optimal_a)
    print(r'The optimal Dirichlet condition is with $T_{end}$ =', optimal_T_end)
    
    return optimal_a, optimal_T_end

opt_dir_cond(1, 0.1, 100, Constant(1.3e-4*3600))


#     solution_list.append(C_T.vector().get_local())

#     for i, solution in enumerate(solution_list):
#         plt.plot(V.tabulate_dof_coordinates(), solution, label=f"a = {a_values[i]}")

# plt.legend()
# plt.xlabel(r'distance [mm]')
# plt.ylabel(r'concentration [mm$^{{-3}}$]')
# plt.title(r'Nitsche Method, $\lambda$ = 100')
# plt.show()

# def optimal_dir_cond(T, dt, D, a):

#     mesh = IntervalMesh(100, 0, 10)

#     num_steps = int(T/dt)

#     V = FunctionSpace(mesh, "CG", 1)

#     C_D = Expression('a*t*(48 - t)', t = 1e-16, a = a, degree = 2) 

#     data = Expression("erfc(x[0]/(2*sqrt(D*t)))*(a*t*(48-t))", a = 1, D = 1.3e-4*3600, t = 1e-16, degree=1)
#     d = project(data, V)

#     C = TrialFunction(V)
#     v = TestFunction(V)

#     # Define initial value
#     C_T = Function(V)
#     # d = Function(V, name="data")
#     C_n = Function(V)

#     F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
#     A, L = lhs(F), rhs(F)

#     def GammaL(x, on_boundary):
#         return near(x[0], 0)

#     bc = DirichletBC(V, C_D, GammaL)

#     J = 0.5*float(dt)*assemble(inner(C_T - data, C_T - data)*dx)

#     t = float(0)

#     J_values = []
    
#     for n in tqdm(range(num_steps)):
#         # Update time
#         t += dt
#         # Update Dirichlet boundary condition
#         C_D.t = t 

#         # Update data function
#         data.t = t
#         d.assign(project(data, V))
        
#         # Solve PDE
#         solve(A == L, C_T, bc)

#         C_n.assign(C_T)

#         # Cost functional 
#         J += 0.5*dt*assemble(inner(C_T - data, C_T - data)*dx) 
#         print("Step:", n, "t:", t, "J:", J)
#         J_values.append(J)

#     control = Control(a)

#     minimized_functional = ReducedFunctional(J, control)

#     # Minimization of the cost function   
#     a_optimal = minimize(minimized_functional, method = 'BFGS', tol = 1e-10, options = {'disp':True})

#     optimal_diffusion = float(a_optimal)

#     print('The optimal coefficient for the Dirichlet condition is:', optimal_diffusion)

#     return optimal_diffusion

# optimal_dir_cond(1, 0.1, Constant(1.3e-4*3600), Constant(1))