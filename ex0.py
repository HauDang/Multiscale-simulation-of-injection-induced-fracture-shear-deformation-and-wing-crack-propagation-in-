import numpy as np
import porepy as pp
import porepy.models.contact_mechanics_biot_model as model

def trisurf( p, t, face = None, infor = None, value = None, vector = None, point = None, show = None, fig = None, grid = None):
    import numpy as np
    if fig is not None:
        grid.clear()
    if show is not None:
        import matplotlib.pyplot as plt
        fig, grid = plt.subplots()
    if t.shape[1] == 3:
        X = [p[t[:,0],0], p[t[:,1],0], p[t[:,2],0], p[t[:,0],0]]
        Y = [p[t[:,0],1], p[t[:,1],1], p[t[:,2],1], p[t[:,0],1]]
        grid.plot(X, Y, 'k-', linewidth = 1)
        if infor is not None:
            cenx = (p[t[:,0],0] + p[t[:,1],0] + p[t[:,2],0])/3
            ceny = (p[t[:,0],1] + p[t[:,1],1] + p[t[:,2],1])/3
            for i in range(t.shape[0]):
                grid.annotate(str(i), (cenx[i], ceny[i]), (cenx[i], ceny[i]), color='blue', fontsize = 14)
            for j in range(p.shape[0]):
                grid.annotate(str(j), (p[j,0], p[j,1]), (p[j,0], p[j,1]), color='red', fontsize = 14)
        if face is not None:
            faccen = ( p[face[:,0],:] + p[face[:,1],:])/2
            for i in range(faccen.shape[0]):
                grid.annotate(str(i), (faccen[i,0], faccen[i,1]), (faccen[i,0], faccen[i,1]), color='m', fontsize = 14)
            
    if t.shape[1] == 6:
        X = [p[t[:,0],0], p[t[:,2],0], p[t[:,4],0], p[t[:,0],0]]
        Y = [p[t[:,0],1], p[t[:,2],1], p[t[:,4],1], p[t[:,0],1]]
        grid.plot(X, Y, 'k-', linewidth = 1)
        if infor is not None:
            cenx = (p[t[:,0],0] + p[t[:,2],0] + p[t[:,4],0])/3
            ceny = (p[t[:,0],1] + p[t[:,2],1] + p[t[:,4],1])/3
            for i in range(t.shape[0]):
                grid.annotate(str(i), (cenx[i], ceny[i]), (cenx[i], ceny[i]), color='blue', fontsize = 14)
            for j in range(p.shape[0]):
                grid.annotate(str(j), (p[j,0], p[j,1]), (p[j,0], p[j,1]), color='red', fontsize = 14) 
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
    if vector is not None:
        plt.quiver(vector[:,0], vector[:,1], vector[:,2], vector[:,3], scale = 1)
    if point is not None:
        grid.plot(point[:,0], point[:,1],'ro')
    if show is not None:
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
        cou = np.max(frac_edges)+1
        
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
        super().__init__(params)
 
        self.tips, self.frac_pts, self.frac_edges = fracture_infor(fracture)
        self.box = box
        self.mesh_args = mesh_args
        
        # Scaling coefficients
        self.scalar_scale = 1 #1e5
        self.length_scale = 1 #100
        self.scalar_source_value = 1

        # Time seconds
        self.time = 0 
        self.end_time = 10 #1*24*60*60
        self.num_step = 3
        self.time_step = self.end_time/self.num_step       

        # solution 
        self.disp = []
        self.pres = []
        self.traction = []
    def prepare_simulation(self):        
        self.create_grid()
        self.set_rock_and_fluid()
        self._set_parameters()
        self._assign_variables()
        self._assign_discretizations()
        self._initial_condition()
        self._discretize()
        self._initialize_linear_solver()
    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to granite and water.
        """
        E = 1.44e10
        nuy = 0.2
          
        self.rock = pp.Granite()
        self.rock.YOUNG_MODULUS = E
        self.rock.POISSON_RATIO = nuy
        self.rock.LAMBDA = E * nuy / ((1 + nuy) * (1 - 2 * nuy))
        self.rock.MU = E / (2 * (1 + nuy))
        self.rock.PERMEABILITY = 2e-13
        self.rock.VISCOSITY = 1.3e-4
        self.rock.DENSITY = 2000
    def _biot_alpha(self, g: pp.Grid) -> float:
        return 0.79
    def _set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self._Nd:
                # Rock parameters
                lam = self.rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = self.rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                # Define boundary condition
                bc = self._bc_type_mechanics(g)
                # BC and source values
                bc_val = self._bc_values_mechanics(g)
                source_val = self._source_mechanics(g)

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

    def _set_scalar_parameters(self) -> None:
        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = 1 * tensor_scale
        mass_weight = 1 * self.scalar_scale
        for g, d in self.gb:
            bc = self._bc_type_scalar(g)
            bc_values = self._bc_values_scalar(g)
            source_values = self._source_scalar(g)

            specific_volume = self._specific_volume(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            alpha = self._biot_alpha(g)
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
    def create_grid(self):
        """ Define a fracture network and domain and create a GridBucket.
        """
        # Domain definition
        network = pp.FractureNetwork2d(self.frac_pts.T, self.frac_edges.T, domain=self.box)
        gb = network.mesh(self.mesh_args)       
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()
        self._Nd = self.gb.dim_max()
        g = self.gb.grids_of_dimension(2)[0]
        p = g.nodes; 
        self.p = p[[0,1],:].T
        self.t = g.cell_nodes().indices.reshape((3, g.num_cells), order='f').T
        return gb
    def _bc_type_mechanics(self, g):
        # dir = displacement 
        # neu = traction
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g)
        
        bc.is_neu[:, north] = True
        bc.is_dir[1, south] = True; #bc.is_neu[0, south] = True
        
        bc.is_dir[0, east] = True; #bc.is_neu[1, east] = True
        
        bc.is_dir[0, west] = True; #bc.is_neu[1, west] = True
        
        return bc
    def _bc_values_mechanics(self, g):
        # Set the boundary values
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))
        values[1, north] = -4e6
        return values.ravel("F")
    def _bc_type_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryCondition(g)  
        return bc 

    def _bc_values_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        return bc_values
    def after_newton_convergence(self, solution, errors, iteration_counter):
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
    
        
mesh_size = 0.01
mesh_args = { "mesh_size_frac": mesh_size, "mesh_size_min": 1 * mesh_size, "mesh_size_bound": 5 * mesh_size } 
params = {"folder_name": "biot_2","convergence_tol": 2e-7,"max_iterations": 20,"file_name": "main_run",}

box = {"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 3}  
fracture1 = np.array([[0.8, 1.5], [1.2, 2]])    
fracture = np.array([fracture1])
fracture = np.array([])


setup = ModelSetup(box, fracture, mesh_args, params)
setup.prepare_simulation()
solver = pp.NewtonSolver(params)

k = 0
while setup.time < setup.end_time:
    setup.time += setup.time_step
    print('step = ', k, ': time = ', setup.time)
    solver.solve(setup)
    dispi = setup.disp[k].reshape((setup.t.shape[0],2))
    presi = setup.pres[k]
    
    # estimating solutions at nodes to plot
    k+= 1
    dispnod =  NN_recovery( dispi, setup.p, setup.t)
    presnod =  NN_recovery( presi, setup.p, setup.t)
    
    dispnod[setup.p[:,0] <= np.finfo(float).eps,0] = 0
    dispnod[setup.p[:,0] >= setup.box['xmax'] - np.finfo(float).eps,0] = 0
    dispnod[setup.p[:,1] <= np.finfo(float).eps,1] = 0
    
    trisurf(setup.p + dispnod*1, setup.t, infor = None, value = dispnod[:,1],  vector = None, point = None, show = 1)
    


trisurf(setup.p + dispnod*1, setup.t, infor = None, value = presnod[:,0],  vector = None, point = None, show = 1)
     
