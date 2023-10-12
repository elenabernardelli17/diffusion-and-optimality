"""
Problem with source term and no Dirichlet boundary condition
The control is source term
"""

from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

data = Expression("erfc(x[0]/(2*sqrt(D*t)))", D = 1.2e-4*3600, t = 0, degree=4)

# Define the diffusion parameter
D = Constant(1.2e-4*3600) #mm^2/h

mesh = IntervalMesh(100, 0, 10)

V = FunctionSpace(mesh, "CG", 1)

dt = Constant(1)
T = 8 # Hours

ctrls = OrderedDict()
t = float(dt)
while t <= T:
    ctrls[t] = Function(V)
    t += float(dt)

# f = Expression('t*(48 - t)', t = 0, degree = 1)
# df = project(Expression('48-2*t', t = 0, degree = 1), V)

f = Function(V, name="source")

def solve_diffusion(ctrls, source_ctrls = None):
    u = TrialFunction(V)
    v = TestFunction(V)

    u_T = Function(V, name="solution")
    d = Function(V, name="data")

    F = ((u - u_T)/dt*v + D*inner(grad(u), grad(v)) - f*v)*dx
    a, L = lhs(F), rhs(F)

    bc = DirichletBC(V, 0, "on_boundary")

    t = float(dt)

    J = 0.5*float(dt)*assemble((u_T - d)**2*dx)
    numerical_solutions = []
    while tqdm(t <= T):
        # Use source_ctrls as the source term f
        if source_ctrls:
            f.assign(source_ctrls[t])
        else:
            f.assign(ctrls[t])

        # df.t = t
        # Update data function
        data.t = t
        d.assign(interpolate(data, V))

        # Solve PDE
        solve(a == L, u_T, bc)

        # Implement a trapezoidal rule
        if t > T - float(dt):
           weight = 0.5
        else:
           weight = 1


        J += weight*float(dt)*assemble((u_T - d)**2*dx) 
        # Update time
        t += float(dt)
    # numerical_solutions.append((V.tabulate_dof_coordinates(), u_T.vector())) 
    # for x, vec in numerical_solutions:
    #     plt.plot(x, vec)
        

    return u_T, d, J

u, d, j = solve_diffusion(ctrls)

alpha = Constant(1e-6)
regularisation = alpha/2*sum([1/dt*(fb-fa)**2*dx for fb, fa in
    zip(list(ctrls.values())[1:], list(ctrls.values())[:-1])])

J = j + assemble(regularisation)

m = [Control(c) for c in ctrls.values()]

rf = ReducedFunctional(J, m)
opt_ctrls = minimize(rf, tol = 1e-2)

# opt_ctrl_values = opt_ctrls
# source_ctrls = OrderedDict()
# for i, t in enumerate(ctrls.keys()):
#     source_ctrls[t] = Function(V, name=f"source_ctrl_{i+1}")
#     source_ctrls[t].assign(opt_ctrl_values[i])

# u2, d2, J2 = solve_diffusion(ctrls, source_ctrls)



# plt.figure()
# time = np. arange(1, T+1, 1)
# for t in time:
#     x = [c(t) for c in opt_ctrls]
#     plt.plot(x)
# legend = ["t=%d"% t for t in time]
# plt.legend(legend)
# plt.ylim(-0.25,1)
# plt.savefig('source_term_opt_t')

plt.figure()
x = [c(1) for c in opt_ctrls]
plt.plot(x)
legend = ['alpha=1e-6, beta=1']
plt.legend(legend)
plt.savefig('source_term_opt_1')








