from __future__ import print_function
from fenics import *
from fenics_adjoint import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from solve_diffusion import solve_dif


# Function to solve my diffusion problem 
def solve_dif_noise(T, dt, N, D):
    
    print ("solving for ", N)
    
    # Define the domain in mm
    mesh = IntervalMesh(N, 0, 10)

    # Define time discretization parameter
    num_steps = int(T/dt)

    # Define the finite element spaces and functions
    V = FunctionSpace(mesh, "CG", 1)
    C = TrialFunction(V)
    v = TestFunction(V)   

    data = Expression("erfc(x[0]/(2*sqrt(D*t)))*(t*(48-t))", D = 1.3e-4*3600, t = 1e-16, degree=1)     

    # Define the left boundary domain
    def GammaL(x, on_boundary):
            return near(x[0], 0)

    # Define Dirichlet buondary condition depending on time with noise
    C_D_noise = Expression('t*(48 - t) + e', t=0, e=0, degree=1)
    bc_noise = DirichletBC(V, C_D_noise, GammaL) 

    # Define initial value
    C_n_noise = Function(V)

    # Define the variational problem to be solved at each time step
    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n_noise*v*dx
    a, L = lhs(F), rhs(F)

    # Solution at final time T
    C_T_noise = Function(V)

    t = 0
    J = 0
    noise_time = np.arange(0, T, 1)
    noise = []
    for n in tqdm(range(num_steps + 1)):

        # Update current time
        t += dt

        # Add noise every 6 mins
        if int(t) in noise_time:
            e = np.random.normal(0, 100)
        else:
            e = 0
        noise.append(e)

        C_D_noise.t = t
        C_D_noise.e = e

        bc_noise = DirichletBC(V, C_D_noise, GammaL)

        # Update data function
        data.t = t
                
        # Compute solution solving the variational problem
        solve(a == L, C_T_noise, bc_noise)
        
        # Assign the ne initial value for next time step
        C_n_noise.assign(C_T_noise)

        # Cost functional
        J += 0.5*dt*assemble(inner(C_T_noise - data, C_T_noise - data)*dx) 
        print("Step:", n, "t:", t, "J:", J)
    
    control = Control(D)

    minimized_functional = ReducedFunctional(J, control)

    # Minimization of the cost function   
    D_optimal = minimize(minimized_functional, method = 'BFGS', tol = 1e-10, options = {'disp':True})

    optimal_diffusion = float(D_optimal)

    print('The optimal diffusion coefficient is:', optimal_diffusion)
    # This function returns the solution at final time T
    return C_T_noise, noise, optimal_diffusion

solve_dif_noise(1, 0.01, 100, Constant(1.3e-4*3600))

exit(-1)

# Final time = 1h, time step = 6mins, D = 1.3e-4*3600 mm^2/h
C_T_noise, noise = solve_dif_noise(24, 0.1, 100, 1.3e-4*3600)
V = C_T_noise.function_space()
plt.plot(V.tabulate_dof_coordinates(), C_T_noise.vector().get_local(), linewidth=2, color = 'red')
plt.xlabel(r'Distance [mm]')
plt.ylabel(r'Concentration [mm$^{{-3}}$]')
plt.title('Final time = 24h, time step = 6mins, D = 1.3e-4*3600 mm^2/h')


plt.figure()
C_T = solve_dif(24, 0.1, 100, 1.3e-4*3600)
C_T_noise, noise = solve_dif_noise(24, 0.1, 100, 1.3e-4*3600)
V = C_T.function_space()
plt.plot(V.tabulate_dof_coordinates(), C_T.vector().get_local(), linewidth=2, label ='Without Noise')
plt.plot(V.tabulate_dof_coordinates(), C_T_noise.vector().get_local(), linewidth=2, label = 'With Noise')
plt.xlabel(r'Distance [mm]')
plt.ylabel(r'Concentration [mm$^{{-3}}$]')
plt.legend()
plt.title('Final time = 24h, time step = 6mins, D = 1.3e-4*3600 mm^2/h')

# Plot the trend of noise
plt.figure()
time_values = np.arange(0, 24 + 0.1, 0.1)
plt.plot(time_values, noise, linewidth=2, color='blue')
plt.xlabel('Time [h]')
plt.ylabel('Noise')
plt.title('Trend of Noise in Dirichlet Condition')
plt.grid(True)
plt.show()


plt.show()