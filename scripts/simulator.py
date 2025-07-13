from dolfin import *
import numpy as np
import os
from scipy.io import loadmat
import shutil 

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}   
parameters["form_compiler"]["quadrature_degree"] = 1


script_dir = os.path.dirname(os.path.abspath(__file__))  # Get scripts folder path
parent_dir = os.path.dirname(script_dir)                 # Go up one level
domain_path = os.path.join(parent_dir, 'inputs', 'domain.xml')  # Go to input folder
data_path = os.path.join(parent_dir, 'inputs', 'bc_displacement.mat')  # Go to input folder

mesh = Mesh(domain_path)
data = loadmat(data_path)


W = VectorFunctionSpace(mesh, 'CG', 1)
p, q, dp = Function(W), TestFunction(W), TrialFunction(W)
u, v, du = Function(W), TestFunction(W), TrialFunction(W)

TS0 = TensorFunctionSpace(mesh, 'DG', 0)

outer_boundary = CompiledSubDomain("abs(x[0]+1.1)<=1.e-3 or abs(x[0]-1.1)<=1.e-3 or \
                                    abs(x[1] +1.1)<=1.e-3 or abs(x[1]- 1.1)<=1.e-3 && on_boundary")

# Load GRF boundary condition data

bc_data = data["bc_data"]  # Shape: (N, 201)
x_values = data["X"].flatten()  # The x-coordinates for the boundary condition


def create_boundary_function(mesh, bc_data, x_values):
    # Create separate boundary functions for x and y directions
    boundary_V = FunctionSpace(IntervalMesh(200, -1.1, 1.1), "CG", 1)
    
    # Create boundary functions for both horizontal and vertical boundaries
    horizontal_function = Function(boundary_V)
    vertical_function = Function(boundary_V)
    
    # Interpolate the data for both functions
    horizontal_function.vector()[:] = np.interp(
        boundary_V.tabulate_dof_coordinates().flatten(), 
        x_values, 
        bc_data
    )
    # For vertical boundaries, we use the same realization
    vertical_function.vector()[:] = np.interp(
        boundary_V.tabulate_dof_coordinates().flatten(), 
        x_values, 
        bc_data
    )
    
    horizontal_function.set_allow_extrapolation(True)
    vertical_function.set_allow_extrapolation(True)
    
    return horizontal_function, vertical_function

class TimeDependentBoundaryExpression(UserExpression):
    def __init__(self, horizontal_function, vertical_function, u_r=0.01, **kwargs):
        super().__init__(**kwargs)
        self.horizontal_function = horizontal_function
        self.vertical_function = vertical_function
        self.t = 0.0
        self.u_r = u_r
        
    def eval(self, values, x):
        scale = self.t * self.u_r
        
        # For left and right boundaries (vertical faces)
        if abs(x[0] + 1.1) < 1e-3 or abs(x[0] - 1.1) < 1e-3:
            # Use vertical function for vertical boundaries
            boundary_val = self.vertical_function(x[1])
            values[0] = boundary_val * scale * x[0]
            values[1] = boundary_val * scale * x[1]
            
        # For top and bottom boundaries (horizontal faces)
        elif abs(x[1] + 1.1) < 1e-3 or abs(x[1] - 1.1) < 1e-3:
            # Use horizontal function for horizontal boundaries
            boundary_val = self.horizontal_function(x[0])
            values[0] = boundary_val * scale * x[0]
            values[1] = boundary_val * scale * x[1]
            
        else:
            values[0] = 0
            values[1] = 0
            
    def value_shape(self):
        return (2,)


bc_phi = []
Cr = 1.e-3
# Variational form
unew, pnew, pnew1 = Function(W), Function(W), Function(W)
u, p = Function(W), Function(W) 
uold, pold = Function(W), Function(W)  

class InitialCondition(UserExpression):
    def eval_cell(self, value, x, ufl_cell):      
        if 0.5<=x[0]<0.55 and abs(x[1])<0.01 :
            value[0] = 0
            value[1] = 1
        elif -0.55<x[0]<= -0.5 and abs(x[1])<0.01 :
            value[0] = 0
            value[1] = -1
        elif -0.55<x[1]<= -0.5 and abs(x[0])<0.01 :
            value[0] = 1
            value[1] = 0
        elif 0.5<=x[1]<= 0.55 and abs(x[0])<0.01 :
            value[0] = -1
            value[1] = 0
        elif 0/(2**0.5)<=x[0]< 0.55/(2**0.5) and abs(x[1]-x[0])<=0.015:
            value[0] = -2**0.5/2
            value[1] = 2**0.5/2
            
        elif -0.55/(2**0.5)<=x[0]<= 0 and abs(x[1]-x[0])<=0.015:
            value[0] = 2**0.5/2
            value[1] = -2**0.5/2
            
        elif -0.55/(2**0.5)<=x[0]<= 0 and abs(x[1]+x[0])<=0.015:
            value[0] = -2**0.5/2
            value[1] = -2**0.5/2
            
        elif 0/(2**0.5)<=x[0]< 0.55/(2**0.5) and abs(x[1]+x[0])<=0.015:
            value[0] = 2**0.5/2
            value[1] = 2**0.5/2  
        
        else:
            value[0] = 0
            value[1] = 0  
    def value_shape(self):
        return (2,)



Gc, l, lmbda, mu, eta_eps =  1.721/ (100e3), 0.015, 0*233.82e3, 1, 1.e-3

def energy(alpha1, alpha0, beta):    
    Energy = mu/2 *(alpha0**2 + alpha1**2 +beta**2 -2)+ h(alpha0*alpha1)                                                                                                                                                                                   
    return Energy

def W0(u):
    I = Identity(len(u))
    F = variable(I + grad(u))   
    C = F.T*F
    C00, C01, C10, C11 = C[0,0], C[0,1], C[1,0], C[1,1]
    alpha1 =  C00**(0.5) 
    beta =  C01 / alpha1 
    alpha0 = (C11 - beta*beta)**(0.5) 
    E = energy(alpha1, alpha0, beta)
    stress = diff(E, F) 
    return [E, stress]
                                                                                                                                                                                              
def W1(u,d): 
    I = Identity(len(u))
    F = variable (I + grad(u) ) 
    d= variable(d)           
    Cr = 1.e-3 
    n1 ,n2 =  d[0]/(sqrt(dot(d,d))) ,  d[1]/(sqrt(dot(d,d)))
    a1, a2 = F[0,0]*n2 - F[0,1]*n1 , F[1,0]*n2 - F[1,1]*n1
    alpha1 =  sqrt(a1**2 + a2**2) 
    alpha0 =  (det(F))/sqrt(a1**2 + a2**2) 
    alpha0_s =   ( ((lmbda*alpha1)**2 +4*mu*lmbda*(alpha1**2)+ 4*mu**2)** (0.5) +\
        lmbda*alpha1)  / (2*(lmbda*(alpha1**2) +mu)) 
    E =   conditional(lt(alpha0, alpha0_s),  energy(alpha1, alpha0, 0), \
          energy(alpha1, alpha0_s, 0) )                            
    stress, dE_dd = diff(E, F) , diff(E, d) 
    return [E, stress, dE_dd]
   
def h(J):
    return (lmbda/2)*(J-1)**2 -mu*ln(J)

def total_energy(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*W0(u)[0] +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 )*\
        conditional(lt(dot(d,d), Cr),0.0, W1(u,d)[0]) +\
        Gc* ( dot(d,d)/(2*l) + (l/2)*inner(grad(d), grad(d)) )              
    return E

def elastic_energy(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 + eta_eps)*W0(u)[0] +\
        (1-(1- conditional(lt(dot(d,d), Cr),0.0, sqrt(dot(d,d))))**2 )*\
        conditional(lt(dot(d,d), Cr),0.0, W1(u,d)[0])              
    return E

        
Pi1 = total_energy(u, pold) * dx          					    
Pi2 = total_energy(unew, p) * dx 

E_du = derivative(Pi1, u, v)   
E_phi = derivative(Pi2, p, q)
 

J_u = derivative(E_du, u, du)    
J_phi  = derivative(E_phi, p, dp)


constraint_l = Constant((0.0, 0))
constraint_u = Constant((1.0, 1))

p_min = interpolate(InitialCondition(), W)
p_max = interpolate(constraint_u, W)

class Problem(OptimisationProblem):
    def __init__(self):
        OptimisationProblem.__init__(self)
    def f(self, x):
        p.vector()[:] = x
        return assemble(Pi2)
    def F(self, b, x):
        p.vector()[:] = x
        assemble(E_phi, tensor=b)
    def J(self, A, x):
        p.vector()[:] = x
        assemble(J_phi, tensor=A)

solver = PETScTAOSolver()
solver.parameters["method"] = "tron"
solver.parameters["monitor_convergence"] = False
solver.parameters["report"] = False
solver.parameters["maximum_iterations"] = 1000

parameters.parse()

def preventHeal(pold, pnew):  #conserves the direction of old crack
    pold_nodal_values = pold.vector()
    pold_array = pold_nodal_values.get_local()
    pnew_nodal_values = pnew.vector() 
    pnew_array = pnew_nodal_values.get_local()
    for i in range(0, len(pold_array), 2):
        pold_mag = sqrt( (pold_array[i])**2 +(pold_array[i+1])**2 )
        pnew_mag = sqrt( (pnew_array[i])**2 +(pnew_array[i+1])**2 )
        if pold_mag > 0.95:
            pnew_array[i], pnew_array[i+1] =  pold_array[i]/pold_mag, \
                pold_array[i+1]/pold_mag        
    pnew3 = Function(W)        
    pnew3.vector()[:] = pnew_array[:]
    return pnew3


def stress_form(u,d):
    E = ((1- conditional(lt(dot(d,d), Cr), 0, sqrt(dot(d,d))))**2 + eta_eps)*W0(u)[1] +\
        (1-(1- conditional(lt(dot(d,d), Cr),0, sqrt(dot(d,d))))**2 )*\
        conditional(lt(dot(d,d), Cr), stress_null, W1(u,d)[1])              
    return E

TS = TensorFunctionSpace(mesh, "CG", 1)
stress = Function(TS)
stress_null = Function(TS)

V = FunctionSpace(mesh, 'CG', 1)
V0 = FunctionSpace(mesh, 'DG', 0)

energy_total = Function(V0)
energy_elastic = Function(V0)

W_deg0 = VectorFunctionSpace(mesh, 'DG', 0)
displacement_integrated = Function(W)
traction_integrated = Function(W_deg0)

def top(x, on_boundary):
    return near(x[1], 1.1) and on_boundary

def bot(x, on_boundary):
    return near(x[1], -1.1) and on_boundary

def left(x, on_boundary):
    return near(x[0], -1.1) and on_boundary

def right(x, on_boundary):
    return near(x[0], 1.1) and on_boundary

boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
AutoSubDomain(top).mark(boundary_subdomains, 1)
AutoSubDomain(left).mark(boundary_subdomains, 2)
AutoSubDomain(bot).mark(boundary_subdomains, 3)
AutoSubDomain(right).mark(boundary_subdomains, 4)
dss = ds(subdomain_data=boundary_subdomains)


def save_simulation_data(folder_name, timestep, displacement_field, phase_field, elastic_energy, fracture_energy,frame):
    
    
    np.save(f'{folder_name}/results_array/displacement/displacement_field_t{timestep:.2f}_frame{frame}.npy', displacement_field)
    np.save(f'{folder_name}/results_array/phase_field/phase_field_t{timestep:.2f}_frame{frame}.npy', phase_field)
    np.save(f'{folder_name}/results_array/elastic_energy/elastic_energy_t{timestep:.2f}_frame{frame}.npy', elastic_energy)
    np.save(f'{folder_name}/results_array/fracture_energy/fracture_energy_t{timestep:.2f}_frame{frame}.npy', fracture_energy)
     
# First, create the base simulation_results directory
base_dir = "../simulation_results"
os.makedirs(base_dir, exist_ok=True)


dof_coords = W.tabulate_dof_coordinates().reshape((-1, 2))

normal_e1 = Constant((1,0))
normal_e2 = Constant((0,1))

tol = 1e-3

# Main simulation loop for multiple realizations
for realization_num in range(1, 101):  # Run for realizations 1 to 10
    try:
        print(f"Starting simulation for realization {realization_num}")
        
        # Create folders for this realization
        folder_name = os.path.join(base_dir, f"realization_{realization_num}")
        
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        
        # Create necessary subdirectories
        os.makedirs(os.path.join(folder_name, 'load_displacement'))
        os.makedirs(os.path.join(folder_name, 'results_snapshots'))
        os.makedirs(os.path.join(folder_name, 'results_array/displacement'))
        os.makedirs(os.path.join(folder_name, 'results_array/phase_field'))
        os.makedirs(os.path.join(folder_name, 'results_array/elastic_energy'))
        os.makedirs(os.path.join(folder_name, 'results_array/fracture_energy'))
        os.makedirs(os.path.join(folder_name, 'results_snapshots/phase_field/paraview'))
        os.makedirs(os.path.join(folder_name, 'results_snapshots/displacement/paraview'))
        
        tens_loading_file = open(os.path.join(folder_name, 'load_displacement/tens_loading.txt'), 'w')
        
        snapshot_folder = os.path.join(folder_name, 'results_snapshots')
        
        bc_realization = bc_data[realization_num-1, :]
        horizontal_function, vertical_function = create_boundary_function(mesh, bc_realization, x_values)
        u_assign = TimeDependentBoundaryExpression(horizontal_function, vertical_function, u_r=0.01, degree=1)
        bc_outer_boundary = DirichletBC(W, u_assign, outer_boundary)
        bc_u = [bc_outer_boundary]
    
    
        
        pvd_phi = File(os.path.join(snapshot_folder, "phase_field/paraview/phi.pvd"))
        pvd_u = File(os.path.join(snapshot_folder, "displacement/paraview/disp.pvd"))
                
        t = 0
        frame = 0
        
        pold.interpolate(InitialCondition())
        p.interpolate(InitialCondition())
        pnew.interpolate(InitialCondition())
        
        u.vector().zero()
        unew.vector().zero()
        uold.vector().zero()
        
        # Save DOF coordinates for this realization
        np.save(os.path.join(folder_name, 'dof_coordinates.npy'), dof_coords)
    
        deltaT  = 0.01  
        while t<= 0.45:
            
            try:
                
                
                t += deltaT
                u_assign.t = t
                iter = 0
                err = 1
                while err > tol:
                    iter += 1
                    solve(E_du ==0, u, bc_u, solver_parameters={'newton_solver':{'maximum_iterations':2000}}) # , form_compiler_parameters=ffc_options, solver_parameters={'newton_solver':{'maximum_iterations':200, 'relaxation_parameter': 0.7, 'linear_solver': 'mumps'}} #solver_disp.solve()
                    unew.assign(u) 
                    solve(E_phi==0, p, bc_phi, solver_parameters={'newton_solver':{'maximum_iterations':2000}})#solver_phi.solve() #solver.solve(Problem(), p.vector(), p_min.vector(), p_max.vector())
                    p_new = preventHeal(pold, p)
                    pnew.assign(p_new)  
                    err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
                    err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
                    print(err_u)
                    print(err_phi)
                    err = max(err_u,err_phi)
                    uold.assign(unew)
                    pold.assign(pnew)
                    p.assign(pnew)
                    
                    # p_min.assign(pnew)
                    if err <= tol:
                        print (' ,Iterations:', iter, ', Total time', t)
                        pnew.rename("d", "crack_vector")
                        unew.rename("u", "displacement")
                        pvd_phi << pnew
                        pvd_u << unew
                        
                        energy_total = project( total_energy(unew, pnew) ,V)
                        energy_elastic = project( elastic_energy(unew, pnew) ,V)
                        
                        
                        displacement_field = unew.vector().get_local()
                        phase_field = pnew.vector().get_local()
                        elastic_energy_field = energy_elastic.vector().get_local()
                        fracture_energy_field = energy_total.vector().get_local()
                        
                        disp_magnitude = sqrt(dot(unew, unew))
                        displacement_integrated = assemble(disp_magnitude * (dss(1) + dss(2) + dss(3) + dss(4)))
                        
                        
                        Stress_tot = project(stress_form(unew,pnew), TS0)
                        
                        traction_top = project(dot(Stress_tot, normal_e2), W)
                        traction_magnitude_top = sqrt(dot(traction_top, traction_top))
                        traction_integrated_top = assemble(traction_magnitude_top * dss(1))
                        
                        # Bottom boundary
                        traction_bot = project(dot(Stress_tot, normal_e2), W)
                        traction_magnitude_bot = sqrt(dot(traction_bot, traction_bot))
                        traction_integrated_bot = assemble(traction_magnitude_bot * dss(3))
                        
                        # Left boundary
                        traction_left = project(dot(Stress_tot, normal_e1), W)
                        traction_magnitude_left = sqrt(dot(traction_left, traction_left))
                        traction_integrated_left = assemble(traction_magnitude_left * dss(2))
                        
                        # Right boundary
                        traction_right = project(dot(Stress_tot, normal_e1), W)
                        traction_magnitude_right = sqrt(dot(traction_right, traction_right))
                        traction_integrated_right = assemble(traction_magnitude_right * dss(4))
                        
                        # Total integrated traction on all boundaries
                        traction_integrated = (traction_integrated_top + traction_integrated_bot + 
                      traction_integrated_left + traction_integrated_right)


                        save_simulation_data( 
                        folder_name=folder_name,
                        timestep=t,
                        displacement_field=displacement_field,
                        phase_field=phase_field,
                        elastic_energy=elastic_energy_field,
                        fracture_energy=fracture_energy_field, 
                        frame=frame
                        )
                        
                        tens_loading_file.write(f"{displacement_integrated:.6f}\t{traction_integrated:.6f}\n")
                        
                        frame += 1
            except Exception as e:
                    print(f"Error in tension phase for realization {realization_num} at time {t}: {str(e)}")
                    raise  # Re-raise to be caught by outer try-except
                    
        print('Tension phase completed')
        tens_loading_file.close()
        
        
        
        tens_unloading_file = open(os.path.join(folder_name, 'load_displacement/tens_unloading.txt'), 'w')
        
        
        
        deltaT  = -0.01  
        while t>= 0.005:
            try:
            
                t += deltaT
                
                u_assign.t = t
                iter = 0
                err = 1
                while err > tol:
                    iter += 1
                    solve(E_du ==0, u, bc_u, solver_parameters={'newton_solver':{'maximum_iterations':2000}}) # , form_compiler_parameters=ffc_options, solver_parameters={'newton_solver':{'maximum_iterations':200, 'relaxation_parameter': 0.7, 'linear_solver': 'mumps'}} #solver_disp.solve()
                    unew.assign(u) 
                    solve(E_phi==0, p, bc_phi, solver_parameters={'newton_solver':{'maximum_iterations':2000}})#solver_phi.solve() #solver.solve(Problem(), p.vector(), p_min.vector(), p_max.vector())
                    p_new = preventHeal(pold, p)
                    pnew.assign(p_new)  
                    err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
                    err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
                    print(err_u)
                    print(err_phi)
                    err = max(err_u,err_phi)
                    uold.assign(unew)
                    pold.assign(pnew)
                    p.assign(pnew)
                    
                    # p_min.assign(pnew)
                    if err <= tol:
                        print (' ,Iterations:', iter, ', Total time', t)
                        pnew.rename("d", "crack_vector")
                        unew.rename("u", "displacement")
                        
                        pvd_phi << pnew
                        pvd_u << unew
                        
                        energy_total = project( total_energy(unew, pnew) ,V)
                        energy_elastic = project( elastic_energy(unew, pnew) ,V)
                        
                        displacement_field = unew.vector().get_local()
                        phase_field = pnew.vector().get_local()
                        elastic_energy_field = energy_elastic.vector().get_local()
                        fracture_energy_field = energy_total.vector().get_local()
                        
                        disp_magnitude = sqrt(dot(unew, unew))
                        displacement_integrated = assemble(disp_magnitude * (dss(1) + dss(2) + dss(3) + dss(4)))
                        
                        
                        Stress_tot = project(stress_form(unew,pnew), TS0)
                        
                        traction_top = project(dot(Stress_tot, normal_e2), W)
                        traction_magnitude_top = sqrt(dot(traction_top, traction_top))
                        traction_integrated_top = assemble(traction_magnitude_top * dss(1))
                        
                        # Bottom boundary
                        traction_bot = project(dot(Stress_tot, normal_e2), W)
                        traction_magnitude_bot = sqrt(dot(traction_bot, traction_bot))
                        traction_integrated_bot = assemble(traction_magnitude_bot * dss(3))
                        
                        # Left boundary
                        traction_left = project(dot(Stress_tot, normal_e1), W)
                        traction_magnitude_left = sqrt(dot(traction_left, traction_left))
                        traction_integrated_left = assemble(traction_magnitude_left * dss(2))
                        
                        # Right boundary
                        traction_right = project(dot(Stress_tot, normal_e1), W)
                        traction_magnitude_right = sqrt(dot(traction_right, traction_right))
                        traction_integrated_right = assemble(traction_magnitude_right * dss(4))
                        
                        # Total integrated traction on all boundaries
                        traction_integrated = (traction_integrated_top + traction_integrated_bot + 
                      traction_integrated_left + traction_integrated_right)
                        
                        save_simulation_data( 
                        folder_name=folder_name,
                        timestep=t,
                        displacement_field=displacement_field,
                        phase_field=phase_field,
                        elastic_energy=elastic_energy_field,
                        fracture_energy=fracture_energy_field, 
                        frame=frame
                        )
                        
                        tens_unloading_file.write(f"{displacement_integrated:.6f}\t{traction_integrated:.6f}\n")
                        
                        
                        frame += 1  # Increment frame counter
             
    
            except Exception as e:
                    print(f"Error in tensile unloading phase for realization {realization_num} at time {t}: {str(e)}")
                    raise  # Re-raise to be caught by outer try-except
                    
        print ('Simulation tensile unloading completed')
        tens_unloading_file.close()
        
        
        comp_loading_file = open(os.path.join(folder_name, 'load_displacement/comp_loading.txt'), 'w')
        
        
        
        while t>= -0.355:
            try:
            
                t += deltaT
                
                u_assign.t = t
                iter = 0
                err = 1
                while err > tol:
                    iter += 1
                    solve(E_du ==0, u, bc_u, solver_parameters={'newton_solver':{'maximum_iterations':2000}}) # , form_compiler_parameters=ffc_options, solver_parameters={'newton_solver':{'maximum_iterations':200, 'relaxation_parameter': 0.7, 'linear_solver': 'mumps'}} #solver_disp.solve()
                    unew.assign(u) 
                    solve(E_phi==0, p, bc_phi, solver_parameters={'newton_solver':{'maximum_iterations':2000}})#solver_phi.solve() #solver.solve(Problem(), p.vector(), p_min.vector(), p_max.vector())
                    p_new = preventHeal(pold, p)
                    pnew.assign(p_new)  
                    err_u = errornorm(unew,uold,norm_type = 'l2',mesh = None)
                    err_phi = errornorm(pnew,pold,norm_type = 'l2',mesh = None)
                    print(err_u)
                    print(err_phi)
                    err = max(err_u,err_phi)
                    uold.assign(unew)
                    pold.assign(pnew)
                    p.assign(pnew)
                    
                    # p_min.assign(pnew)
                    if err <= tol:
                        print (' ,Iterations:', iter, ', Total time', t)
                        pnew.rename("d", "crack_vector")
                        unew.rename("u", "displacement")
                        
                        pvd_phi << pnew
                        pvd_u << unew
                        
                        energy_total = project( total_energy(unew, pnew) ,V)
                        energy_elastic = project( elastic_energy(unew, pnew) ,V)
                        
                        displacement_field = unew.vector().get_local()
                        phase_field = pnew.vector().get_local()
                        elastic_energy_field = energy_elastic.vector().get_local()
                        fracture_energy_field = energy_total.vector().get_local()
                        
                        disp_magnitude = sqrt(dot(unew, unew))
                        displacement_integrated = assemble(disp_magnitude * (dss(1) + dss(2) + dss(3) + dss(4)))
                        
                        
                        Stress_tot = project(stress_form(unew,pnew), TS0)
                        
                        traction_top = project(dot(Stress_tot, normal_e2), W)
                        traction_magnitude_top = sqrt(dot(traction_top, traction_top))
                        traction_integrated_top = assemble(traction_magnitude_top * dss(1))
                        
                        # Bottom boundary
                        traction_bot = project(dot(Stress_tot, normal_e2), W)
                        traction_magnitude_bot = sqrt(dot(traction_bot, traction_bot))
                        traction_integrated_bot = assemble(traction_magnitude_bot * dss(3))
                        
                        # Left boundary
                        traction_left = project(dot(Stress_tot, normal_e1), W)
                        traction_magnitude_left = sqrt(dot(traction_left, traction_left))
                        traction_integrated_left = assemble(traction_magnitude_left * dss(2))
                        
                        # Right boundary
                        traction_right = project(dot(Stress_tot, normal_e1), W)
                        traction_magnitude_right = sqrt(dot(traction_right, traction_right))
                        traction_integrated_right = assemble(traction_magnitude_right * dss(4))
                        
                        # Total integrated traction on all boundaries
                        traction_integrated = (traction_integrated_top + traction_integrated_bot + 
                      traction_integrated_left + traction_integrated_right)
                        
                        save_simulation_data( 
                        folder_name=folder_name,
                        timestep=t,
                        displacement_field=displacement_field,
                        phase_field=phase_field,
                        elastic_energy=elastic_energy_field,
                        fracture_energy=fracture_energy_field, 
                        frame=frame
                        )
                        
                        
                        comp_loading_file.write(f"{displacement_integrated:.6f}\t{traction_integrated:.6f}\n")
                        
                        
                        frame += 1  # Increment frame counter
             
    
            except Exception as e:
                    print(f"Error in compression phase for realization {realization_num} at time {t}: {str(e)}")
                    raise  # Re-raise to be caught by outer try-except
                    
        print ('Simulation Compression completed')
    
        comp_loading_file.close()
    
    except Exception as e:
        print(f"Skipping realization {realization_num} due to error: {str(e)}")
        # Close any open files before continuing to next realization
        try:
            tens_loading_file.close()
        except:
            pass
        try:
            tens_unloading_file.close()
        except:
            pass
        try:
            comp_loading_file.close()
        except:
            pass
        continue  # Move to next realization

print('All simulations completed')