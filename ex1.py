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
 
        tips, frac_pts, frac_edges = fracture_infor(fracture)
        self.box = box
        self.tips = tips
        self.frac_pts = frac_pts
        self.frac_edges = frac_edges
        self.mesh_args = mesh_args
        
        # Scaling coefficients
        self.scalar_scale = 1 #1e5
        self.length_scale = 1 #100
        self.scalar_source_value = 1

        # Time seconds
        self.time = 0 
        self.time_step = pp.YEAR/12
        self.end_time = self.time_step * 4
        self.time_index = 0

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
    def _biot_alpha(self, g: pp.Grid) -> float:
        return 0.5
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
        p = g.nodes; self.p = p[[0,1],:].T
        self.t = g.cell_nodes().indices.reshape((3, g.num_cells), order='f').T
        return gb
    def _source_mechanics(self, g):
        return np.zeros(g.num_cells * self._Nd)

    def _source_scalar(self, g):
        if g.dim == self._Nd:
            values = np.zeros(g.num_cells)
        else:
            values = self.scalar_source_value * np.ones(g.num_cells)
        return values

    def _bc_type_mechanics(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, north + south, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        
        return bc

    def _bc_type_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryCondition(g)
        # Define boundary condition on faces
        bc.is_neu[west] = True 
        bc.is_neu[east] = True   
        return bc # displacement

    def _bc_values_mechanics(self, g):
        # Set the boundary values
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))

        values[0, south] = 0
        values[1, south] = 0
        values[0, north] = 0.005*np.sin(self.time_index)
        values[1, north] = 0.002
        return values.ravel("F")
    def _bc_values_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        # bc_values[east] = 0
        # bc_values[west] = 0
        return bc_values
    def _friction_coefficient(self):
        pass
        return

    def after_newton_convergence(self, solution, errors, iteration_counter):
        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        dispi, presi, tractioni = self.export_results()
        self.disp.append(dispi.reshape((self.t.shape[0],2), order='f'))
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
# box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
# fracture1 = np.array([[0.4, 0.5], [0.6, 0.5]])
# fracture = np.array([fracture1])

fracture1 = np.array([[0.2, 0.7], [0.5, 0.7], [0.8, 0.65]])
fracture2 = np.array([[1, 0.3], [1.8,0.4]])
fracture3 = np.array([[0.2, 0.3], [0.6, 0.25]])
fracture4 = np.array([[1.0, 0.4], [1.7, 0.85]])
fracture5 = np.array([[1.5, 0.65], [2.0, 0.55]])
fracture6 = np.array([[1.5, 0.05], [1.4, 0.25]])
fracture = np.array([fracture1, fracture2, fracture3, fracture4, fracture5, fracture6])
box = {'xmin': 0, 'ymin': 0, 'xmax': 2, 'ymax': 1}


mesh_args = { "mesh_size_frac": mesh_size, 
              "mesh_size_min": 1 * mesh_size, 
              "mesh_size_bound": 3 * mesh_size } 
params = {
    "nl_convergence_tol": 1e-8,
    "max_iterations": 50,
}

setup = ModelSetup(box, fracture, mesh_args, params)

pp.run_time_dependent_model(setup, params)
setup.time_index
for i in range(3):
    dispi = setup.disp[i]
    presi = setup.pres[i]
    dispnod =  NN_recovery( dispi, setup.p, setup.t)
    presnod =  NN_recovery( presi, setup.p, setup.t)
    
    trisurf(setup.p+dispnod*0.001, setup.t, face = None, index = None, value = presnod)

trisurf(setup.p, setup.t, face = None, index = None, value = None)

