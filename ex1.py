import numpy as np
import porepy as pp
import porepy.models.contact_mechanics_biot_model as model

def trisurf(p, t, face = None, index = None, value = None):
    import matplotlib.pyplot as plt
    fig, grid = plt.subplots()
    X = [p[t[:,0],0], p[t[:,1],0], p[t[:,2],0], p[t[:,0],0]]
    Y = [p[t[:,0],1], p[t[:,1],1], p[t[:,2],1], p[t[:,0],1]]
    grid.plot(X, Y, 'k-', linewidth = 1)
    if index is not None:
        cenx = (p[t[:,0],0] + p[t[:,1],0] + p[t[:,2],0])/3
        ceny = (p[t[:,0],1] + p[t[:,1],1] + p[t[:,2],1])/3
        for i in range(t.shape[0]):
            grid.annotate(str(i), (cenx[i], ceny[i]), color='blue', fontsize = 14)
        for j in range(p.shape[0]):
            grid.annotate(str(j), (p[j,0], p[j,1]), color='red', fontsize = 14)
        if face is not None:
            faccen = ( p[face[:,0],:] + p[face[:,1],:])/2
            for i in range(faccen.shape[0]):
                grid.annotate(str(i), (faccen[i,0], faccen[i,1]), color='m', fontsize = 20)
    if value is not None:
        if len(value.shape) == 1:
            value = value.reshape(len(value),1)
        name_color_map = 'jet'
        x = p[:,0]
        # y = p[:,1]
        # z = value[:,0]
        # plt.tricontourf(x,y,t,z,1000,cmap = name_color_map)
        # plt.colorbar()
        if t.shape[1] == 3:
            x = p[:,0]
            y = p[:,1]
            z = value[:,0]
            plt.tricontourf(x,y,t,z,1000,cmap = name_color_map)
            plt.colorbar()
 
        if t.shape[1] == 6:
            snode = np.max(t[:,[0, 2, 4]]) + 1
            x = p[0:snode,0]
            y = p[0:snode,1]
            z = value[0:snode,0]
            tt = t[:,[0, 2, 4]]
            plt.tricontourf(x,y,tt,z,500,cmap = name_color_map)
            plt.colorbar()    
    plt.show()
def fracture_infor(fracture):
    def tip_edge_fracture(fracture):
        cou = 0
        edges = np.array([0, 0]).reshape(1,2)
        tips = np.array([0, 0]).reshape(1,2)
        tips_fraci = np.vstack((fracture[0,:], fracture[-1,:]))
        tips = np.concatenate((tips, tips_fraci), axis = 0)
        for j in range(fracture.shape[0] - 1):
            indedg = np.array([[cou, cou+1]])
            edges = np.concatenate((edges, indedg), axis = 0)
            cou = cou + 1
        tips = np.delete(tips, 0, 0)
        edges = np.delete(edges, 0, 0)  
        return tips, edges
    
    cou = 0; frac_pts = np.array([0, 0]).reshape(1,2); tips = np.array([0, 0]).reshape(1,2); frac_edges = np.array([0, 0]).reshape(1,2)
    for i in range(len(fracture)):
        frac_pts = np.concatenate((frac_pts,fracture[i]), axis = 0)
        tipsi, edgei = tip_edge_fracture(fracture[i])
        tips = np.concatenate((tips, tipsi), axis = 0)
        frac_edges = np.concatenate((frac_edges,edgei + cou), axis = 0)
        cou = np.max(edgei)+1
        
    tips = np.delete(tips, 0, 0)
    frac_edges = np.delete(frac_edges, 0, 0)     
    frac_pts = np.delete(frac_pts, 0, 0)     
    
    return tips, frac_pts, frac_edges
def NN_recovery( values, p, t):
    ''' Natural neighbor interpolation 
        Approximate values at nodes from average cells value'''
    if len(values.shape) == 1:
        values = values.reshape(len(values),1)
    valnod = np.zeros((p.shape[0],values.shape[1]))
    for i in range(values.shape[1]):
        indmat = np.zeros((t.shape[0],np.max(t) + 1  ))
        valmat = np.zeros((t.shape[0],np.max(t) + 1  ))         
        for e in range(t.shape[0] ):
            valmat[e,t[e,:]] = values[e,i]
            indmat[e,t[e,:]] = 1
        X = p[t,0]
        Y = p[t,1]
        Ae = polygon_area(X,Y)
        Ae = Ae.reshape(len(Ae),1)
        vale = np.dot(np.transpose(valmat),Ae)/np.dot(np.transpose(indmat),Ae)
        valnod[:,i] = vale[:,0]# values at interpolation points
    return valnod
def polygon_area(x, y): 
    '''https://stackoverrun.com/vi/q/6706068'''
    correction = x[:,-1] * y[:,0] - y[:,-1]* x[:,0]
    main_area = np.sum(x[:,:-1] * y[:,1:], axis = 1) - np.sum(y[:,:-1] * x[:,1:], axis = 1)
    return 0.5*np.abs(main_area + correction)
class ModelSetup(model.ContactMechanicsBiot):
    def __init__(self, box, fracture, mesh_args, params):
        """ Set arguments for mesh size (as explained in other tutorials)
        and the name fo the export folder.
        """
        # super().__init__(mesh_args, folder_name)
        # params kemur nú með folder_name
        super().__init__(params)
 
        # Set additional case specific fields
        # For now I take this out but can be used to set paremeters later on (see main.py (THM))
        tips, frac_pts, frac_edges = fracture_infor(fracture)
        self.box = box
        self.tips = tips
        self.frac_pts = frac_pts
        self.frac_edges = frac_edges
        self.mesh_args = mesh_args
        
       
        # Fix names used for the variables:
        # Names for mechanics variables are set in a super class
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable
        
        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

        self.subtract_fracture_pressure = True

        # Scaling coefficients
        # The use of this is not mature. Be careful!
        self.scalar_scale = 1 #1e5
        self.length_scale = 1 #100
        self.scalar_source_value = 1

        # Time 
        self.time = pp.YEAR # seconds
        self.time_step = self.time/2 #1500 * pp.DAY # obs scales with length scale
        self.end_time = self.time_step * 10
        self.step = 0

        self.initial_aperture = 1e-3 #/ self.length_scale
        # Dirichlet boundary condition for pressure
        self.s_0 = 1 * pp.MEGA / self.scalar_scale
        # solution 
        self.disp = []
        self.pres = []
        self.traction = []
    def before_newton_loop(self):
        self.convergence_status = False
        self._iteration = 0
        self.set_parameters()
    def set_parameters(self):
        self.set_scalar_parameters()
        self.set_mechanics_parameters()
    def prepare_simulation(self):

        
        self.create_grid()
        self.set_rock_and_fluid()
        self.set_parameters()
        self._assign_variables()
        self._assign_discretizations()
        self.initial_condition()
        self._discretize()
        self._initialize_linear_solver()

        g_max = self._nd_grid()
        self.viz = pp.Exporter(
            g_max, file_name="mechanics", folder_name=self.viz_folder_name
        )
    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to granite and water.
        """
        self.rock = pp.Granite()
        self.rock.FRICTION_COEFFICIENT = 0.5
        self.rock.PERMEABILITY = 1e-16

    def biot_alpha(self):
        return 1
    
    def create_grid(self):
        """ Define a fracture network and domain and create a GridBucket. 
        
        This setup has two fractures inside the unit cell.
        
        The method also calls a submethod which sets injection points in 
        one of the fractures.
        
        """
        # Domain definition
        network = pp.FractureNetwork2d(self.frac_pts.T, self.frac_edges.T, domain=self.box)
        gb = network.mesh(self.mesh_args)       
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()
        self._Nd = self.gb.dim_max()
        g = self.gb.grids_of_dimension(2)[0]
        p = g.nodes; self.p = p[[0,1],:].T
        self.t = g.cell_nodes().indices.reshape((3, g.num_cells), order='f').T
        return gb
    def source_mechanics(self, g):
        return np.zeros(g.num_cells * self._Nd)

    def source_scalar(self, g):
        if g.dim == self._Nd:
            values = np.zeros(g.num_cells) + self.time_step*self.step
        else:
            values = self.scalar_source_value * np.ones(g.num_cells)
        return values

    def bc_type_mechanics(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, north + south, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        
        return bc

    def bc_type_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryCondition(g)
        # Define boundary condition on faces
        bc.is_neu[west] = True 
        bc.is_neu[east] = True   
        return bc # displacement

    def bc_values_mechanics(self, g):
        # Set the boundary values
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))

        values[0, south] = 0
        values[1, south] = 0
        values[0, north] = 1e6 #self.time_step
        values[1, north] = 0
        return values.ravel("F")
    def bc_values_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        # bc_values[east] = 0
        # bc_values[west] = 0
        return bc_values
    
    def set_mechanics_parameters(self):
        
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                rock = self.rock
                lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                # Define boundary condition
                bc = self.bc_type_mechanics(g)
                # BC and source values
                bc_val = self.bc_values_mechanics(g)
                source_val = self.source_mechanics(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": C,
                        "time_step": self.time_step,
                        "biot_alpha": self._biot_alpha(g),
                    },
                )

            elif g.dim == self._Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction, "time_step": self.time_step},
                )

        for _, d in gb.edges():
            mg: pp.MortarGrid = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)
    def set_scalar_parameters(self):
        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = 1 * tensor_scale
        mass_weight = 1 * self.scalar_scale
        for g, d in self.gb:
            bc = self.bc_type_scalar(g)
            bc_values = self.bc_values_scalar(g)
            source_values = self.source_scalar(g)

            specific_volume = self._specific_volume(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            alpha = self.biot_alpha()
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight * specific_volume,
                    "biot_alpha": alpha,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "time_step": self.time_step,
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            a_l = self._aperture(g_l)
            # Take trace of and then project specific volumes from g_h
            v_h = (
                mg.primary_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self._specific_volume(g_h)
            )
            # Division by a/2 may be thought of as taking the gradient in the normal
            # direction of the fracture.
            normal_diffusivity = kappa * 2 / (mg.secondary_to_mortar_avg() * a_l)
            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_h
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"normal_diffusivity": normal_diffusivity},
            )
    def initial_condition(self, time=0):
        """
        Initial guess for Newton iteration, scalar variable and bc_values (for time
        discretization).
        """
        super()._initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            initial_scalar_value = np.zeros(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})
            if g.dim == self._Nd:
                bc_values = self.bc_values_mechanics(g)
                mech_dict = {"bc_values": bc_values}
                d[pp.STATE].update({self.mechanics_parameter_key: mech_dict})

        for _, d in self.gb.edges():
            mg = d["mortar_grid"]
            initial_value = np.zeros(mg.num_cells)
            d[pp.STATE][self.mortar_scalar_variable] = initial_value
    def after_newton_convergence(self, solution, errors, iteration_counter):
        self.step = self.step+1
        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        dispi, presi, tractioni = self.export_results()
        self.disp.append(dispi)
        self.pres.append(presi)
        self.traction.append(tractioni)
        return

    def export_results(self):
        """
        Save displacement jumps and number of iterations for visualisation purposes. 
        These are written to file and plotted against time in Figure 4.
        """     
        g = self.gb.grids_of_dimension(2)[0]
        data = self.gb.node_props(g)
        disp = data[pp.STATE]["u"]
        pres = data[pp.STATE]["p"]
        
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]
        
        traction = matrix_dictionary['stress'] * disp + matrix_dictionary["grad_p"] * pres

        return disp, pres, traction
    
        
mesh_size = 0.05
box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
fracture1 = np.array([[0.4, 0.5], [0.6, 0.5]])
fracture = np.array([fracture1])

mesh_args = { "mesh_size_frac": mesh_size, 
              "mesh_size_min": 1 * mesh_size, 
              "mesh_size_bound": 3 * mesh_size } 
params = {
    "nl_convergence_tol": 1e-8,
    "max_iterations": 50,
}

setup = ModelSetup(box, fracture, mesh_args, params)

pp.run_time_dependent_model(setup, params)

for i in range(3):
    dispi = setup.disp[i]
    presi = setup.pres[i]
    presnod =  NN_recovery( presi, setup.p, setup.t)
    
    trisurf(setup.p, setup.t, face = None, index = None, value = presnod)
























