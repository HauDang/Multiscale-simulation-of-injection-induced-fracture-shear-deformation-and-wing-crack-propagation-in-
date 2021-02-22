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
    # if t.shape[1] == 3:
    #     X = [p[t[:,0],0], p[t[:,1],0], p[t[:,2],0], p[t[:,0],0]]
    #     Y = [p[t[:,0],1], p[t[:,1],1], p[t[:,2],1], p[t[:,0],1]]
    #     grid.plot(X, Y, 'k-', linewidth = 1)
    #     if infor is not None:
    #         cenx = (p[t[:,0],0] + p[t[:,1],0] + p[t[:,2],0])/3
    #         ceny = (p[t[:,0],1] + p[t[:,1],1] + p[t[:,2],1])/3
    #         for i in range(t.shape[0]):
    #             grid.annotate(str(i), (cenx[i], ceny[i]), (cenx[i], ceny[i]), color='blue', fontsize = 14)
    #         for j in range(p.shape[0]):
    #             grid.annotate(str(j), (p[j,0], p[j,1]), (p[j,0], p[j,1]), color='red', fontsize = 14)
    #     if face is not None:
    #         faccen = ( p[face[:,0],:] + p[face[:,1],:])/2
    #         for i in range(faccen.shape[0]):
    #             grid.annotate(str(i), (faccen[i,0], faccen[i,1]), (faccen[i,0], faccen[i,1]), color='m', fontsize = 14)
            
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
def adjustmesh(g, tips, dc = None):
    from numpy import matlib as mb
    def projection(A,B,M):
        dis1 = np.sqrt(sum((M - A)**2))
        dis2 = np.sqrt(sum((M - B)**2))
        if dis1 < np.finfo(float).eps*1E3:
            N = A; flag = 1; dis = 0
        elif dis2 < np.finfo(float).eps*1E3:
            N = B; flag = 1; dis = 0
        else:
            AB = np.sqrt(sum((B - A)**2))
            tanvec = (B - A)/AB
            a1, b1 = -tanvec[1], tanvec[0]
            c1 = -a1*A[0] - b1*A[1]
            
            a2, b2 = tanvec[0], tanvec[1]
            c2 = -a2*M[0] - b2*M[1]
            if (a1 == 0 and b1 == 0) or (a2 == 0 and b2 == 0):
                print('something wrong in intersection. please check')
                xn, yn = A
            elif a1 == 0 and b2 == 0:
                xn, yn = -c2/a2, -c1/b1
            elif b1 == 0 and a2 == 0:
                xn, yn = -c1/a1, -c2/b2
            elif a1 == 0:
                xn, yn = (-c2 + b2*c1/b1)/a2, -c1/b1
            elif b1 == 0:
                xn, yn = -c1/a1, (-c2 + a2*c1/a1)/b2
            elif a2 == 0:
                xn, yn = (-c1 + b1*c2/b2)/a1, -c2/b2
            elif b2 == 0:
                xn, yn = -c2/a2, (-c1 + a1*c2/a2)/b1
            else:
                xn, yn = -(c1/b1 - c2/b2)/(a1/b1 - a2/b2), -(c1/a1 - c2/a2)/(b1/a1 - b2/a2)
            N = np.array(([xn, yn]))
            dis = np.sqrt(sum((M - N)**2))
            if dis < np.finfo(float).eps*1E3:
                if abs(np.sqrt(sum((A - N)**2)) + np.sqrt(sum((B - N)**2)) - AB) < np.finfo(float).eps*1E3:
                    flag = 2 # N belong to AB
                else:
                    flag = 0 # N dose not belong to AB
            else:
                dir1 = np.sign(N - A)
                dir2 = np.sign(N - B)
                if (dir1[0] == 0 and dir1[1] == 0) or (dir2[0] == 0 and dir2[1] == 0):
                    flag = 1 # N == A or N == B
                elif dir1[0] == -dir2[0] and dir1[1] == -dir2[1]:
                    flag = 2 # N belong to AB
                else:
                    flag = 0
        return N, flag, dis
    if dc is None:
        dc = 1E-4
    p = g.nodes
    t = g.cell_nodes().indices.reshape((3, g.num_cells), order='f').T
    p = p[[0,1],:].T
    # nodaro = p[g.get_boundary_nodes(),:]
    
    # px = np.tile(p[:,0].reshape(p.shape[0],1),(1,p.shape[0])) 
    # py = np.tile(p[:,1].reshape(p.shape[0],1),(1,p.shape[0]))  
    
    px = mb.repmat(p[:,0].reshape(p.shape[0],1),1,p.shape[0])
    py = mb.repmat(p[:,1].reshape(p.shape[0],1),1,p.shape[0])
    
    ds = np.sqrt( (px - px.T)**2 + (py - py.T)**2 )
    idi, idj = np.where(ds < np.finfo(float).eps*1E5)
    index = np.where(idi != idj)[0]
    fraind = np.zeros(shape = (len(index),2))
    fraind[:,0] = idi[index]
    fraind[:,1] = idj[index]
    fraind = np.int32(np.unique(np.sort(fraind, axis = 1), axis = 0))
    tips_nod = []
    if tips is not None:
        for i in range(tips.shape[0]):
            tip = tips[i,:]
            id1 = np.where(np.isclose(p[:,0],tip[0]))[0]
            id2 = np.where(np.isclose(p[:,1],tip[1]))[0]
            tipind = np.intersect1d(id2,id1)[0]
            tips_nod.append(tipind)
            
        face_ind = np.reshape(g.face_nodes.indices, (g.dim, -1), order='F').T
        face_nor = g.face_normals.T; face_nor = face_nor[:,[0,1]]
        frac_fac = np.where(g.tags['fracture_faces'])[0]  
        node_adj_old = []
        for i in range(len(frac_fac)):
            index = face_ind[frac_fac[i],:]
            ele = np.intersect1d(np.where(t == index[0])[0], np.where(t == index[1])[0])
            index0 = np.setdiff1d(t[ele,:], index)[0]
            A, B = p[index,:]
            M = p[index0,:]
            N,_,_= projection(A,B,M)
            normi = (N - M); normi = normi/np.sqrt(sum(normi**2))
            node_adj = np.setdiff1d(np.setdiff1d(index,tips_nod),node_adj_old)
            p[node_adj,:] = p[node_adj,:] - normi*dc/2
            node_adj_old = np.concatenate((node_adj_old,node_adj))
            
    return p, t
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
        self.end_time = 120*60*60
        self.num_step = 5
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
        E = 10e9
        nuy = 0.2
        permeability = np.array([[2e-15, 0],[0, 1e-15]])
        viscosity = 1
        
        permeability_frac = 1e-10
        viscosity_frac = 1
        
        bulk_fluid = 1/(5e-9)
        bulk_soild = 0.8*E/3/(1 - 2*nuy)
        

        self.LAMBDA = E * nuy / ((1 + nuy) * (1 - 2 * nuy))
        self.MU = E / (2 * (1 + nuy))
        
        self.COMPRESSIBILITY = (0.75-0.2)/bulk_soild + 0.2/bulk_fluid
        self.DIFFUSIVITY = permeability/viscosity
        self.DIFFUSIVITY_frac = permeability_frac/viscosity_frac
        
        
        self.MASS_DENSITY = 2e3
        
    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self._Nd:
            aperture *= 0.005
        return aperture    
    def _biot_alpha(self, g: pp.Grid) -> float:
        return 0.75
    def _set_friction_coefficient(self, g: pp.Grid) -> np.ndarray:
        """The friction coefficient is uniform, and equal to 1."""
        return np.ones(g.num_cells)*0
    def _set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self._Nd:
                # Rock parameters
                lam = self.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = self.MU * np.ones(g.num_cells) / self.scalar_scale
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
        for g, d in self.gb:
            bc = self._bc_type_scalar(g)
            bc_values = self._bc_values_scalar(g)
            source_values = self._source_scalar(g)

            specific_volume = self._specific_volume(g)
           
            diffusivity = pp.SecondOrderTensor( kxx = self.DIFFUSIVITY[0,0]*specific_volume*np.ones(g.num_cells),
                                                kyy = self.DIFFUSIVITY[1,1]*specific_volume*np.ones(g.num_cells),
                                                kzz = None, 
                                                kxy = self.DIFFUSIVITY[1,0]*specific_volume*np.ones(g.num_cells))
            alpha = self._biot_alpha(g)
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": self.COMPRESSIBILITY * specific_volume,
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
            normal_diffusivity = self.DIFFUSIVITY_frac * 2 / (mg.secondary_to_mortar_avg() * a_l)
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
    def _source_mechanics(self, g):
        Fm = np.zeros(g.num_cells * self._Nd) 
        Fm[1::2] = self.MASS_DENSITY*9.8*g.cell_volumes*0
        return Fm

    def _source_scalar(self, g):
        values = np.zeros(g.num_cells) + 1000*9.8*g.cell_volumes*0
        return values

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
        # p = g.nodes; 
        # self.p = p[[0,1],:].T
        # self.t = g.cell_nodes().indices.reshape((3, g.num_cells), order='f').T
        self.p, self.t = adjustmesh(g, self.tips)
        return gb
    def _bc_type_mechanics(self, g):
        # dir = displacement 
        # neu = traction
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g)
        
        bc.is_neu[:, north] = True
        bc.is_dir[:, south] = True
        bc.is_dir[0, east] = True
        bc.is_dir[0, west] = True
        
        # frac_face = g.tags["fracture_faces"]
        # bc.is_dir[:, frac_face] = True
        
        return bc
    def _bc_values_mechanics(self, g):
        # Set the boundary values
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))
        values[1, north] = -20e6*g.face_areas[north]
        return values.ravel("F")
    def _bc_type_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc = pp.BoundaryCondition(g)
        bc.is_dir[north] = True
        # frac_face = g.tags["fracture_faces"]
        # bc.is_neu[frac_face] = True
        return bc 

    def _bc_values_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self._domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        bc_values[north] = 0*g.face_areas[north]
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
    
        
mesh_size = 0.1
mesh_args = { "mesh_size_frac": mesh_size, "mesh_size_min": 1 * mesh_size, "mesh_size_bound": 5 * mesh_size } 
params = {"folder_name": "biot_2","convergence_tol": 2e-7,"max_iterations": 20,"file_name": "main_run",}

box = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 20}  
fracture1 = np.array([[5, 4], [5, 16]])    
fracture = np.array([fracture1])
# fracture = np.array([])


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
    
    trisurf(setup.p + dispnod*100, setup.t, infor = None, value = dispnod[:,0],  vector = None, point = None, show = 1)
    trisurf(setup.p + dispnod*100, setup.t, infor = None, value = dispnod[:,1],  vector = None, point = None, show = 1)
    trisurf(setup.p + dispnod*100, setup.t, infor = None, value = presnod[:,0],  vector = None, point = None, show = 1)


  