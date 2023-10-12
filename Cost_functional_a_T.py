from fenics import *
from fenics_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import erfc


def optimal_dir_cond_1(T, dt, N, D, a, T_end):

    mesh = IntervalMesh(N, 0, 10)

    num_steps = int(T/dt)

    V = FunctionSpace(mesh, "CG", 1)

    C_D = Expression('a*t*(T_end - t)', t = 1e-16, a = Constant(a), T_end = Constant(T_end), degree =1) 

    data = Expression("erfc(x[0]/(2*sqrt(D*t)))*(t*(48-t))", D = 1.3e-4*3600, t = 1e-16, degree=1)

    C = TrialFunction(V)
    v = TestFunction(V)

    # Define initial value
    C_T = Function(V)
    C_n = Function(V)

    F = C*v*dx + dt*D*dot(grad(C), grad(v))*dx - C_n*v*dx
    A, L = lhs(F), rhs(F)

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
        solve(A == L, C_T, bc)

        C_n.assign(C_T)

        # Cost functional 
        J += 0.5*dt*assemble(inner(C_T - data, C_T - data)*dx) 
        print("Step:", n, "t:", t, "J:", J)
        J_values.append(J)

    return J

a_values = np.arange(0.9, 1.7, 0.1)
T_end_values = np.arange(39, 60, 1)

a_T_matrix = np.zeros((len(a_values), len(T_end_values)))
print(np.shape(a_T_matrix))

for i, a in enumerate(a_values):
    for j, T_end in enumerate(T_end_values):
        a_T_matrix[i,j] = optimal_dir_cond_1(1, 0.001, 100, 1.3e-4*3600, a, T_end)

min_value = print(np.min(a_T_matrix))

min_indices = np.unravel_index(np.argmin(a_T_matrix), a_T_matrix.shape)

min_a = a_values[min_indices[0]]
min_T_end = T_end_values[min_indices[1]]

print('a value:', min_a, 'T_end value:', min_T_end)


# plt.contour(T_end_values, a_values, a_T_matrix, cmap='viridis', levels=100)
# plt.colorbar(label=r'Cost Functional')
# plt.xlabel(r'T_end')
# plt.ylabel(r'a')
# plt.title('Cost Functional')
# plt.show()


# Definisci la scala dei colori
# cmap = 'viridis'

# # Crea il grafico imshow con personalizzazioni
# plt.figure() 
# plt.imshow(a_T_matrix, extent=[T_end_values.min(), T_end_values.max(), a_values.min(), a_values.max()],
#            aspect='auto', cmap=cmap, origin='lower')

# # Aggiungi una barra dei colori
# cbar = plt.colorbar()
# cbar.set_label('Valori')

# # Definisci i livelli di contour separati
# contour_levels = np.arange(40, 60, 1)  # Personalizza questi valori

# Crea il grafico imshow con personalizzazioni
plt.figure(figsize=(8, 6))  # Imposta la dimensione della figura
plt.contourf(T_end_values, a_values, a_T_matrix, levels=60, cmap='viridis')

# Aggiungi una barra dei colori
cbar = plt.colorbar()
cbar.set_label('Valori')

# Aggiungi delle etichette agli assi
plt.xlabel(r'$T_{end}$')
plt.ylabel('a')

# Aggiungi un titolo al grafico
plt.title(r'Optimal Dirichlet Condition with parameters a and $T_{end}$ after 1 hour, $\Delta$t = 36 seconds')

plt.show()