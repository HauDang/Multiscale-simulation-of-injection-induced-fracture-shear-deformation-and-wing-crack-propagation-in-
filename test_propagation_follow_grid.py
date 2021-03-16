import numpy as np
import time
import porepy as pp
import porepy.models.contact_mechanics_biot_model as parent_model
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
        X = [p[t[:,0],0], p[t[:,1],0], p[t[:,2],0], p[t[:,3],0], p[t[:,4],0], p[t[:,5],0], p[t[:,0],0]]
        Y = [p[t[:,0],1], p[t[:,1],1], p[t[:,2],1], p[t[:,3],1], p[t[:,4],1], p[t[:,5],1], p[t[:,0],1]]
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
        dc = 5E-3
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
class ModelSetup(pp.ContactMechanicsBiot, pp.ConformingFracturePropagation):
    def __init__(self, box, fracture, mesh_args, params):
        """ Set arguments for mesh size (as explained in other tutorials)
        and the name fo the export folder.
        """
        super().__init__(params)        
        
        self.fracture = fracture
 
        self.tips, self.frac_pts, self.frac_edges = fracture_infor(self.fracture)
        self.initial_aperture = 0.001
        self.box = box
        self.mesh_args = mesh_args
        
        # Scaling coefficients
        self.scalar_scale = 1 #1e5
        self.length_scale = 1 #100
        self.scalar_source_value = 1

        # Time seconds
        self.time = 0 
        self.end_time = 120*60*60
        self.num_step = 10
        self.time_step = self.end_time/self.num_step       

        # solution 
        self.disp = []
        self.pres = []
        self.traction = []
    def prepare_simulation(self):        
        self.create_grid()
        self.set_rock_and_fluid()
        self.set_parameters()
        self.assign_variables()
        self.assign_discretizations()
        self.initial_condition()
        self.discretize()
        self.initialize_linear_solver()
    def update_discretize(self):        
        self.set_parameters()
        self.assign_variables()
        self.assign_discretizations()
        self.discretize()
        self.initialize_linear_solver()
    def before_newton_loop(self):
        self.convergence_status = False
        self._iteration = 0
        
    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to granite and air.
        """
        E = 40e9
        nuy = 0.2
        density = 2700
        permeability = 1e-8 * pp.DARCY
        porosity = 0.01
        biot = 0.75
        bulk_solid = E/3/(1 - 2*nuy)
        
        viscosity = 1.81e-5
        bulk_fluid = 101e3
        
        permeability_frac = 1e-12
        viscosity_frac = 1
        
              
        self.YOUNG = E
        self.POISSON = nuy
        self.KIC = 0
        self.LAMBDA = E * nuy / ((1 + nuy) * (1 - 2 * nuy))
        self.MU = E / (2 * (1 + nuy))
        self.DENSITY = density
        
        self.COMPRESSIBILITY = (biot - porosity)/bulk_solid + porosity/bulk_fluid
        self.DIFFUSIVITY = np.array([[permeability/viscosity, 0],[0, permeability/viscosity]])
        self.DIFFUSIVITY_frac = permeability_frac/viscosity_frac
        
        
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
    def set_mechanics_parameters(self) -> None:
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

    def set_scalar_parameters(self) -> None:
        for g, d in self.gb:
            bc = self._bc_type_scalar(g)
            bc_values = self._bc_values_scalar(g)
            source_values = self._source_scalar(g)

            specific_volume = self.specific_volume(g)
           
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
            # k: pp.SecondOrderTensor = self.scalar_parameter_key["second_order_tensor"]
        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            a_l = self._aperture(g_l)
            # Take trace of and then project specific volumes from g_h
            v_h = (
                mg.primary_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self.specific_volume(g_h)
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
        Fm[1::2] = self.DENSITY*9.8*g.cell_volumes
        return Fm

    def _source_scalar(self, g):
        values = np.zeros(g.num_cells)
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
        g2d = self.gb.grids_of_dimension(2)[0]
        self.max_pro = np.min(g2d.face_areas)
        # p = g.nodes; 
        # self.p = p[[0,1],:].T
        # self.t = g.cell_nodes().indices.reshape((3, g.num_cells), order='f').T
        self.p, self.t = adjustmesh(g2d, self.tips)
        return gb
    def _bc_type_mechanics(self, g):
        # dir = displacement 
        # neu = traction
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g)
        
        bc.is_neu[:, north] = True
        bc.is_dir[:, south] = True
        # bc.is_dir[0, east] = True
        # bc.is_dir[0, west] = True
        
        frac_face = g.tags["fracture_faces"]
        bc.is_dir[:, frac_face] = True
        
        return bc
    def _bc_values_mechanics(self, g):
        # Set the boundary values
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))
        values[1, north] = -2e6*g.face_areas[north]
        values[0, north] = -1e3*g.face_areas[north]*0
        return values.ravel("F")
    def _bc_type_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc = pp.BoundaryCondition(g)
        bc.is_dir[north] = True
        # frac_face = g.tags["fracture_faces"]
        # bc.is_neu[frac_face] = True
        return bc 

    def _bc_values_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        bc_values[north] = 0*g.face_areas[north]
        return bc_values
    def after_newton_convergence(self, solution, errors, iteration_counter):
        """Propagate fractures if relevant. Update variables and parameters
        according to the newly calculated solution.
        
        """
        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        gb = self.gb
        
        
        solution_2d = solution[:self.assembler.full_dof[0] + self.assembler.full_dof[1]]
        solution_1d = solution[self.assembler.full_dof[0] + self.assembler.full_dof[1]:]

        # We export the converged solution *before* propagation:
        self.update_all_apertures(to_iterate=True)
        self.export_step()
        # NOTE: Darcy fluxes were updated in self.after_newton_iteration().
        # The fluxes are mapped to the new geometry (and fluxes are assigned for
        # newly formed faces) by the below call to self._map_variables().

        # Propagate fractures:
        #   i) Identify which faces to open in g_h
        #   ii) Split faces in g_h
        #   iii) Update g_l and the mortar grid. Update projections.
        # self.evaluate_propagation()
        g2d = gb.grids_of_dimension(2)[0]
        g1d = gb.grids_of_dimension(1)[0]
        
        d1 = gb.node_props(g2d)
        disp_cells = d1[pp.STATE]["u"].reshape((self.t.shape[0],2))
        pres_cells = d1[pp.STATE]["p"]
   
        disp_nodes =  NN_recovery( disp_cells, self.p, self.t)
        # trisurf( self.p + disp_nodes*5e2 , self.t, face = None, infor = None, value = disp_nodes[:,1], vector = None, point = None, show = 1)
        
        newfrac, tips0, p6, t6, disp6 = fracture_analysis.evaluate_propagation(self.YOUNG, self.POISSON, self.KIC, g2d, self.p, self.t, self.fracture, self.tips, self.max_pro, disp_nodes)
        dic_split = re_meshing.remesh_at_tip(gb, self.fracture, newfrac[0])  
       
        pp.contact_conditions.set_projections(gb)

        p2, t2 = adjustmesh(g2d, tips0)
        mapping = np.asarray(pp.intersections.triangulations(p2.T, self.p.T, t2.T, self.t.T) )
        pres2 = np.zeros( shape = (t2.shape[0]))
        disp2 = np.zeros( shape = (t2.shape[0],2))
        for e in range(t2.shape[0]):
            ind = np.where(mapping[:,0] == e)[0]
            concel = np.int32( mapping[ind,1] )
            disp2[e,:] = np.array( [sum(disp_cells[concel,0]*mapping[ind,2])/sum(mapping[ind,2]), 
                                    sum(disp_cells[concel,1]*mapping[ind,2])/sum(mapping[ind,2]) ])
            pres2[e] = sum(pres_cells[concel]*mapping[ind,2])/sum(mapping[ind,2])
        
        gb.node_props(g2d)[pp.STATE]["u"] = disp2.reshape((g2d.num_cells*2,1))[:,0]
        gb.node_props(g2d)[pp.STATE]["p"] = pres2
        self.assembler.full_dof[0] = g2d.num_cells*2
        self.assembler.full_dof[1] = g2d.num_cells
        self.gb = gb

        solution = np.concatenate((disp2.reshape((g2d.num_cells*2,1))[:,0], pres2, solution_1d))
        self.assembler.distribute_variable(solution)
        self.update_discretize()
        
        pp.propagate_fracture.propagate_fractures(gb, dic_split)
        self.propagated_fracture = True

        if self.propagated_fracture:
            # Update parameters and discretization

            for g, d in gb:
                if g.dim < self.Nd - 1:
                    # Should be really careful in this situation. Fingers crossed.
                    continue

                # Transfer information on new faces and cells from the format used
                # by self.evaluate_propagation to the format needed for update of
                # discretizations (see Discretization.update_discretization()).
                # TODO: This needs more documentation.
                new_faces = d.get("new_faces", np.array([], dtype=np.int))
                split_faces = d.get("split_faces", np.array([], dtype=np.int))
                modified_faces = np.hstack((new_faces, split_faces))
                update_info = {
                    "map_cells": d["cell_index_map"],
                    "map_faces": d["face_index_map"],
                    "modified_cells": d.get("new_cells", np.array([], dtype=np.int)),
                    "modified_faces": d.get("new_faces", modified_faces),
                }
                d["update_discretization"] = update_info

            # Map variables after fracture propagation. Also initialize variables
            # for newly formed cells, faces and nodes.
            # Also map darcy fluxes and time-dependent boundary values (advection
            # and the div_u term in poro-elasticity).
            new_solution = self._map_variables(solution)

            # Update apertures: Both state (time step) and iterate.
            self.update_all_apertures(to_iterate=False)
            self.update_all_apertures(to_iterate=True)

            # Set new parameters.
            self.set_parameters()
            # For now, update discretizations will do a full rediscretization
            # TODO: Replace this with a targeted rediscretization.
            # We may want to use some of the code below (after return), but not all of
            # it.
            self._minimal_update_discretization()
        else:
            # No updates to the solution
            new_solution = solution

        # Finally, use super's method to do updates not directly related to
        # fracture propgation
        super().after_newton_convergence(new_solution, errors, iteration_counter)

        # self.adjust_time_step()

        # Done!
        self.fracture = np.array( [np.concatenate(( newfrac[0][1,:].reshape(1,2), self.fracture[0]), axis = 0)] )
        self.tips = tips0
        
        dispi, presi, tractioni = self.export_results()
        self.disp.append(dispi)
        self.pres.append(presi)
        self.traction.append(tractioni)
        return
    # def after_newton_convergence(self, solution, errors, iteration_counter):
    #     """Propagate fractures if relevant. Update variables and parameters
    #     according to the newly calculated solution.
        
    #     """
    #     self.assembler.distribute_variable(solution)
    #     self.convergence_status = True
    #     gb = self.gb
        
        
    #     # We export the converged solution *before* propagation:
    #     self.update_all_apertures(to_iterate=True)
    #     self.export_step()
    #     # NOTE: Darcy fluxes were updated in self.after_newton_iteration().
    #     # The fluxes are mapped to the new geometry (and fluxes are assigned for
    #     # newly formed faces) by the below call to self._map_variables().

    #     # Propagate fractures:
    #     #   i) Identify which faces to open in g_h
    #     #   ii) Split faces in g_h
    #     #   iii) Update g_l and the mortar grid. Update projections.
    #     # self.evaluate_propagation()
    #     split_face = 5
    #     g_1d = gb.grids_of_dimension(1)[0]
    #     dic_split = {}
    #     d = { g_1d: split_face }
    #     dic_split.update(d)  
        
    #     g = self.gb.grids_of_dimension(2)[0]
    #     p = g.nodes; p = p[[0,1],:].T
        
    #     newfrac = []
    #     newfrac.append( np.array([p[0,:].reshape(1,2), p[67,:].reshape(1,2)]) )


    #     # split_face = 133
    #     # g_1d = gb.grids_of_dimension(1)[0]
    #     # dic_split = {}
    #     # d = { g_1d: split_face }
    #     # dic_split.update(d)  
        
    #     # g = self.gb.grids_of_dimension(2)[0]
    #     # p = g.nodes; p = p[[0,1],:].T
        
    #     # newfrac = []
    #     # newfrac.append( np.array([p[68,:].reshape(1,2), p[41,:].reshape(1,2)]) )

        
    #     pp.propagate_fracture.propagate_fractures(gb, dic_split)
    #     self.propagated_fracture = True

    #     if self.propagated_fracture:
    #         # Update parameters and discretization

    #         for g, d in gb:
    #             if g.dim < self.Nd - 1:
    #                 # Should be really1 careful in this situation. Fingers crossed.
    #                 continue

    #             # Transfer information on new faces and cells from the format used
    #             # by self.evaluate_propagation to the format needed for update of
    #             # discretizations (see Discretization.update_discretization()).
    #             # TODO: This needs more documentation.
    #             new_faces = d.get("new_faces", np.array([], dtype=np.int))
    #             split_faces = d.get("split_faces", np.array([], dtype=np.int))
    #             modified_faces = np.hstack((new_faces, split_faces))
    #             update_info = {
    #                 "map_cells": d["cell_index_map"],
    #                 "map_faces": d["face_index_map"],
    #                 "modified_cells": d.get("new_cells", np.array([], dtype=np.int)),
    #                 "modified_faces": d.get("new_faces", modified_faces),
    #             }
    #             # d["update_discretization"] = update_info

    #         # Map variables after fracture propagation. Also initialize variables
    #         # for newly formed cells, faces and nodes.
    #         # Also map darcy fluxes and time-dependent boundary values (advection
    #         # and the div_u term in poro-elasticity).
    #         new_solution = self._map_variables(solution)

    #         # Update apertures: Both state (time step) and iterate.
    #         self.update_all_apertures(to_iterate=False)
    #         self.update_all_apertures(to_iterate=True)

    #         # Set new parameters.
    #         self.set_parameters()
    #         # For now, update discretizations will do a full rediscretization
    #         # TODO: Replace this with a targeted rediscretization.
    #         # We may want to use some of the code below (after return), but not all of
    #         # it.
    #         self._minimal_update_discretization()
    #     else:
    #         # No updates to the solution
    #         new_solution = solution

    #     # Finally, use super's method to do updates not directly related to
    #     # fracture propgation
    #     super().after_newton_convergence(new_solution, errors, iteration_counter)

    #     # self.adjust_time_step()

    #     # Done!
    #     self.fracture = np.array( [np.concatenate(( newfrac[0][1,:].reshape(1,2), self.fracture[0]), axis = 0)] )
    #     dispi, presi, tractioni = self.export_results()
    #     self.disp.append(dispi)
    #     self.pres.append(presi)
    #     self.traction.append(tractioni)
    #     return
    def update_all_apertures(self, to_iterate=True):
        """
        To better control the aperture computation, it is done for the entire gb by a
        single function call. This also allows us to ensure the fracture apertures
        are updated before the intersection apertures are inherited.
        The aperture of a fracture is
            initial aperture + || u_n ||
        """
        gb = self.gb
        for g, d in gb:

            apertures = np.ones(g.num_cells)
            if g.dim == (self.Nd - 1):
                # Initial aperture

                apertures *= self.initial_aperture
                # Reconstruct the displacement solution on the fracture
                g_h = gb.node_neighbors(g)[0]
                data_edge = gb.edge_props((g, g_h))
                if pp.STATE in data_edge:
                    u_mortar_local = self.reconstruct_local_displacement_jump(
                        data_edge,
                        d["tangential_normal_projection"],
                        from_iterate=to_iterate,
                    )
                    # Magnitudes of normal components
                    # Absolute value to avoid negative volumes for non-converged
                    # solution (if from_iterate is True above)
                    apertures += np.absolute(u_mortar_local[-1])

            if to_iterate:
                pp.set_iterate(
                    d,
                    {"aperture": apertures.copy(), "specific_volume": apertures.copy()},
                )
            else:
                state = {
                    "aperture": apertures.copy(),
                    "specific_volume": apertures.copy(),
                }
                pp.set_state(d, state)

        for g, d in gb:
            parent_apertures = []
            num_parent = []
            if g.dim < (self.Nd - 1):
                for edges in gb.edges_of_node(g):
                    e = edges[0]
                    g_h = e[0]

                    if g_h == g:
                        g_h = e[1]

                    if g_h.dim == (self.Nd - 1):
                        d_h = gb.node_props(g_h)
                        if to_iterate:
                            a_h = d_h[pp.STATE][pp.ITERATE]["aperture"]
                        else:
                            a_h = d_h[pp.STATE]["aperture"]
                        a_h_face = np.abs(g_h.cell_faces) * a_h
                        mg = gb.edge_props(e)["mortar_grid"]
                        # Assumes g_h is primary
                        a_l = (
                            mg.mortar_to_secondary_avg()
                            * mg.primary_to_mortar_avg()
                            * a_h_face
                        )
                        parent_apertures.append(a_l)
                        num_parent.append(
                            np.sum(mg.mortar_to_secondary_int().A, axis=1)
                        )
                    else:
                        raise ValueError("Intersection points not implemented in 3d")
                parent_apertures = np.array(parent_apertures)
                num_parents = np.sum(np.array(num_parent), axis=0)

                apertures = np.sum(parent_apertures, axis=0) / num_parents

                specific_volumes = np.power(
                    apertures, self.Nd - g.dim
                )  # Could also be np.product(parent_apertures, axis=0)
                if to_iterate:
                    pp.set_iterate(
                        d,
                        {
                            "aperture": apertures.copy(),
                            "specific_volume": specific_volumes.copy(),
                        },
                    )
                else:
                    state = {
                        "aperture": apertures.copy(),
                        "specific_volume": specific_volumes.copy(),
                    }
                    pp.set_state(d, state)

        return apertures
    def _minimal_update_discretization(self):
        # NOTE: Below here is an attempt at local updates of the discretization
        # matrices. For now, these are replaced by a full discretization at the
        # begining of each time step.

        # EK: Discretization is a pain, because of the flux term.
        # The advective term needs an updated (expanded faces) flux term,
        # to compute this, we first need to expand discretization of the
        # pressure diffusion terms.
        # It should be possible to do something smarter here, perhaps compute
        # fluxes before splitting, then transfer numbers and populate with other
        # values. Or something else.
        gb = self.gb

        t_0 = time.time()

        g_max = gb.grids_of_dimension(gb.dim_max())[0]
        grid_list = gb.grids_of_dimension(gb.dim_max() - 1).tolist()
        grid_list.append(g_max)

        data = gb.node_props(g_max)[pp.DISCRETIZATION_MATRICES]

        flow = {}
        for key in data["flow"]:
            flow[key] = data["flow"][key].copy()

        mech = {}
        for key in data["mechanics"]:
            mech[key] = data["mechanics"][key].copy()

        self.discretize_biot(update_after_geometry_change=False)

        for e, _ in gb.edges_of_node(g_max):
            grid_list.append((e[0], e[1], e))

        filt = pp.assembler_filters.ListFilter(
            variable_list=[self.scalar_variable, self.mortar_scalar_variable],
            term_list=[self.scalar_coupling_term],
            grid_list=grid_list,
        )
        self.assembler.discretize(filt=filt)

        grid_list = gb.grids_of_dimension(gb.dim_max() - 1).tolist()
        filt = pp.assembler_filters.ListFilter(
            term_list=["diffusion", "mass", "source"],
            variable_list=[self.scalar_variable],
            grid_list=grid_list,
        )
        # self.assembler.update_discretization(filt=filt)
        self.assembler.discretize(filt=filt)

        edge_list = []
        for e, _ in self.gb.edges():
            edge_list.append(e)
            edge_list.append((e[0], e[1], e))
        if len(edge_list) > 0:
            filt = pp.assembler_filters.ListFilter(grid_list=edge_list)
            self.assembler.discretize(filt=filt)

        # Finally, discretize terms on the lower-dimensional grids. This can be done
        # in the traditional way, as there is no Biot discretization here.
        for dim in range(0, self.Nd):
            grid_list = self.gb.grids_of_dimension(dim)
            if len(grid_list) > 0:
                filt = pp.assembler_filters.ListFilter(grid_list=grid_list)
                self.assembler.discretize(filt=filt)

        # logger.info("Rediscretized in {} s.".format(time.time() - t_0))


    def export_results(self):
        """
        Save displacement jumps and number of iterations for visualisation purposes. 
        These are written to file and plotted against time in Figure 4.
        """     
        g = self.gb.grids_of_dimension(2)[0]
        self.p, self.t = adjustmesh(g, self.tips)
        data = self.gb.node_props(g)
        disp = data[pp.STATE]["u"]
        pres = data[pp.STATE]["p"]
        
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]
        
        traction = matrix_dictionary['stress'] * disp + matrix_dictionary["grad_p"] * pres

        return disp, pres, traction
    def assemble_and_solve_linear_system(self, tol):
        import scipy.sparse.linalg as spla

        A, b = self.assembler.assemble_matrix_rhs()
        if self.linear_solver == "direct":
            sol = spla.spsolve(A, b)
            disp_step = sol[:self.assembler.full_dof[0]].reshape((self.t.shape[0],2))
            return sol
        elif self.linear_solver == "pyamg":
            raise NotImplementedError("Not that far yet")
    
        
mesh_size = 0.5
mesh_args = { "mesh_size_frac": mesh_size, "mesh_size_min": 1 * mesh_size, "mesh_size_bound": 5 * mesh_size } 
params = {"folder_name": "biot_2","convergence_tol": 2e-7,"max_iterations": 20,"file_name": "main_run",}

box = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 20}  
fracture1 = np.array([[4.0, 8], [6.0, 12]])    
fracture = np.array([fracture1])

setup = ModelSetup(box, fracture, mesh_args, params)
setup.prepare_simulation()
solver = pp.NewtonSolver(params)
g2d = setup.gb.grids_of_dimension(2)[0]
p, t = adjustmesh( g2d, setup.tips)    
face =  g2d.face_nodes.indices.reshape((2, g2d.num_faces), order='f').T   
trisurf( p , t, face = None, infor = None, value = None, vector = None, point = None, show = 1)



k = 0


import re_meshing
import fracture_analysis

while setup.time < setup.end_time:
    setup.time += setup.time_step
    print('step = ', k, ': time = ', setup.time)
    
    # g2d = setup.gb.grids_of_dimension(2)[0]
    # p, t = adjustmesh( g2d, setup.tips)
    # face =  g2d.face_nodes.indices.reshape((2, g2d.num_faces), order='f').T   
    # trisurf( p , t, face = face, infor = 1, value = None, vector = None, point = None, show = 1)
    
    solver.solve(setup)
    
    g2d = setup.gb.grids_of_dimension(2)[0]
    p, t = adjustmesh( g2d, setup.tips)    
    face =  g2d.face_nodes.indices.reshape((2, g2d.num_faces), order='f').T   
    disp = setup.disp[k].reshape((t.shape[0],2))
    disp_nodes =  NN_recovery( disp, p, t)
    trisurf( p + disp_nodes*5e2 , t, face = face, infor = 1, value = disp_nodes[:,1], vector = None, point = None, show = 1)
                      
    k+= 1
    
# disp_nodes =  NN_recovery( disp_step, self.p, self.t)    
# trisurf( self.p + disp_nodes*1e2 , self.t, face = None, infor = None, value = disp_nodes[:,1], vector = None, point = None, show = 1)
# trisurf( p , t, face = fn, infor = 1, value = None, vector = None, point = None, show = 1)
    