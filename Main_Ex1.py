import numpy as np
# np.seterr(divide='raise')
import porepy as pp
import mixedmode_fracture_analysis as analysis
class ModelSetup(pp.ContactMechanicsBiot, pp.ConformingFracturePropagation):
    def __init__(self, box, fracture, mesh_args, params):
        """ Set arguments for mesh size (as explained in other tutorials)
        and the name fo the export folder.
        """
        super().__init__(params)        
        
        self.fracture = fracture
        self.initial_fracture = fracture
        self.GAP = 5e-3
 
        self.tips, self.frac_pts, self.frac_edges = analysis.fracture_infor(self.fracture)
        self.initial_aperture = 0.001
        self.box = box
        self.mesh_args = mesh_args
        
        # Scaling coefficients
        self.scalar_scale = 1 #1e5
        self.length_scale = 1 #100
        self.scalar_source_value = 1

        # Time seconds
        self.time = 0 
        self.end_time = 365*24*60*60
        self.time_step = 24*60*60 # 1 day  
        self.glotim = []

        # solution 
        self.disp = []
        self.pres = []
        self.traction = []
        self.stored_disp = []
        self.stored_pres = []
        # geometry
        self.nodcoo = []
        self.celind = []
        self.facind = []
        self.farcoo = []
        # Propagation criterion
        self.pro_cri = False
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
        self.YOUNG = 40e9
        self.POISSON = 0.2
        self.BIOT = 0.8
        self.POROSITY = 1E-2
        self.COMPRESSIBILITY = 4.0E-10
        self.PERMEABILITY = 1e-15
        self.VISCOSITY = 1
        self.FRICTION = 0.5
        self.FLUID_DENSITY = 1E3

        self.BULK = self.YOUNG/3/(1 - 2*self.POISSON)
        self.LAMBDA = self.YOUNG*self.POISSON / ((1 + self.POISSON) * (1 - 2 * self.POISSON))
        self.MU = self.YOUNG/ (2 * (1 + self.POISSON))
        self.material = dict([('YOUNG', self.YOUNG), ('POISSON', self.POISSON), ('KIC', 540000) ])      
    def porosity(self, g) -> float:
        if g.dim == 2:
            return self.POROSITY
        else:
            return 1.0
    def biot_alpha(self, g) -> np.ndarray:
        if g.dim == 2:
            return self.BIOT
        else:
            return 1.0
    def aperture(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the aperture of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["aperture"]
        else:
            return self.gb.node_props(g)[pp.STATE]['iterate']['aperture']
   
    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self._Nd:
            aperture *= self.GAP
        return aperture    
    def specific_volume(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the specific volume of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["specific_volume"]
        else:
            return self.gb.node_props(g)[pp.STATE]["specific_volume"]    
    def _set_friction_coefficient(self, g: pp.Grid) -> np.ndarray:
        """The friction coefficient is uniform, and equal to 1."""
        return np.ones(g.num_cells) * self.FRICTION
    def set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == 2:
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
                        "biot_alpha": self.biot_alpha(g),
                    },
                )

            elif g.dim == 1:
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
            if g.dim == 2:
                mass_weight = self.COMPRESSIBILITY * self.porosity(g) + (self.BIOT - self.porosity(g)) / self.BULK * specific_volume
            else:
                mass_weight = self.COMPRESSIBILITY * self.porosity(g) * specific_volume
           
            alpha = self.biot_alpha(g)
            
            
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "biot_alpha": alpha,
                    "time_step": self.time_step,
                    "ambient_dimension": self.Nd,
                    "source": source_values,
                },
            )
        self.set_vector_source()
        self.set_permeability_from_aperture()
    def set_vector_source(self):
        for g, d in self.gb:
            grho = (pp.GRAVITY_ACCELERATION * self.FLUID_DENSITY )
            gr = np.zeros((self.Nd, g.num_cells))
            gr[self.Nd - 1, :] = -grho
            d[pp.PARAMETERS][self.scalar_parameter_key]["vector_source"] = gr.ravel("F")
        for e, data_edge in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            params_l = self.gb.node_props(g1)[pp.PARAMETERS][self.scalar_parameter_key]
            mg = data_edge["mortar_grid"]
            grho = (
                mg.secondary_to_mortar_avg()
                * params_l["vector_source"][self.Nd - 1 :: self.Nd]
            )
            a = mg.secondary_to_mortar_avg() * self.aperture(g1)
            gravity = np.zeros((self.Nd, mg.num_cells))
            gravity[self.Nd - 1, :] = grho * a / 2

            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"vector_source": gravity.ravel("F")},
            )
    def set_permeability_from_aperture(self):
        """
        Cubic law in fractures, rock permeability in the matrix.
        """
        # Viscosity has units of Pa s, and is consequently divided by the scalar scale.
        viscosity = self.VISCOSITY
        gb = self.gb
        key = self.scalar_parameter_key
        for g, d in gb:
            specific_volume = self.specific_volume(g)
            if g.dim == 1:
                # Use cubic law in fractures. First compute the unscaled
                # permeability
                apertures = self.aperture(g, from_iterate=True)
                k = np.power(apertures, 2) / 12 / viscosity
                d[pp.PARAMETERS][key]["perm_nu"] = k
                # Multiply with the cross-sectional area, which equals the apertures
                # for 2d fractures in 3d
                kxx = k * specific_volume
            else:
                kxx = self.PERMEABILITY/self.VISCOSITY * np.ones(g.num_cells)
            K = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][key]["second_order_tensor"] = K

        # Normal permeability inherited from the neighboring fracture g_l
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            g_l, g_h = gb.nodes_of_edge(e)
            data_l = gb.node_props(g_l)
            a = self.aperture(g_l, True)
            V = self.specific_volume(g_l, True)
            # V_h = mg.primary_to_mortar_avg() * np.abs(g_h.cell_faces) * self.specific_volume(g_h)
            V_h = self.specific_volume(g_h, True)
            # We assume isotropic permeability in the fracture, i.e. the normal
            # permeability equals the tangential one
            k_s = data_l[pp.PARAMETERS][key]["second_order_tensor"].values[0, 0]
            # Division through half the aperture represents taking the (normal) gradient
            kn = mg.secondary_to_mortar_int() * np.divide(k_s, a * V / 2)
            tr = np.abs(g_h.cell_faces)
            V_j = mg.primary_to_mortar_avg() * tr * V_h
            kn = kn * V_j
            pp.initialize_data(mg, d, key, {"normal_diffusivity": kn})
    def _source_mechanics(self, g):
        Fm = np.zeros(g.num_cells * self._Nd) 

        return Fm

    def _source_scalar(self, g):
        values = np.zeros(g.num_cells)
        if g.dim == 1:
            inject = (g.cell_centers[0,:] <= 2.1) * (g.cell_centers[0,:] >= 1.9) * (g.cell_centers[1,:] <= 2.01) * (g.cell_centers[1,:] >= 1.99)
            values[inject] = g.cell_volumes[inject]*0.001
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
        self.min_face = np.min(g2d.face_areas)
        self.min_cell = np.min(g2d.cell_volumes)
        self.p, self.t = analysis.adjustmesh(g2d, self.tips, self.GAP)
        self.update_all_apertures()
        return gb
    def _bc_type_mechanics(self, g):
        # dir = displacement 
        # neu = traction
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g)
        
        bc.is_neu[:, north] = True
        # bc.is_dir[1, north] = True
        bc.is_dir[:, south] = True
        
        # bc.is_dir[:, north] = True
        # bc.is_dir[:, south] = True
        
        frac_face = g.tags["fracture_faces"]
        bc.is_dir[:, frac_face] = True
        
        return bc
    def _bc_values_mechanics(self, g):
        # Set the boundary values
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))
        values[1, north] = -2e6*g.face_areas[north]
        # values[0, north] = 2e6*g.face_areas[north]
        return values.ravel("F")
    def _bc_type_scalar(self, g):
        bc = pp.BoundaryCondition(g)
        return bc

    def _bc_values_scalar(self, g):
        bc_values = np.zeros(g.num_faces)
        return bc_values
    def p_dir_faces(self, g):
        """
        We prescribe Dirichlet value at the fractures.
        No-flow for the matrix.
        """
        if g.dim == 2:
            return np.empty(0, dtype=int)
        else: 
            dir_scalar = (g.cell_centers[0,:] <= 2.1) * (g.cell_centers[0,:] >= 1.9) * (g.cell_centers[1,:] <= 2.01) * (g.cell_centers[1,:] >= 1.99)
            return dir_scalar
    def after_newton_convergence(self, solution, errors, iteration_counter):
        """Propagate fractures if relevant. Update variables and parameters
        according to the newly calculated solution.
        
        """
        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        self.save_mechanical_bc_values()
        
        # gb = self.gb
        g2d = self.gb.grids_of_dimension(2)[0]
        data = self.gb.node_props(g2d)
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
        # disp_cells = solution[:self.assembler.full_dof[0]].reshape((self.t.shape[0],2))
        # pres_cells = solution[self.assembler.full_dof[0] : self.assembler.full_dof[0] + self.assembler.full_dof[1]]
        
        disp_cells = data[pp.STATE]["u"].reshape((self.t.shape[0],2))
        pres_cells = data[pp.STATE]["p"]
   
        disp_nodes =  analysis.NN_recovery( disp_cells, self.p, self.t)
        pres_nodes =  analysis.NN_recovery( pres_cells, self.p, self.t)
        
        tips, frac_pts, frac_edges = analysis.fracture_infor(self.fracture)

        pmod, tmod = analysis.adjustmesh(g2d, tips, self.GAP) 
        
        pref, tref = analysis.refinement( pmod, tmod, self.fracture, tips, self.min_cell, self.min_face, self.GAP)
        
        keq, newfrac, tips0, p6, t6, disp6 = analysis.evaluate_propagation(self.material, pref, tref, self.p, self.t, 
                                                                      self.initial_fracture, self.fracture, tips, 
                                                                      self.min_face, disp_nodes, self.GAP)
        print(keq)
        if len(newfrac) > 0:
            solution_1d = solution[self.assembler.full_dof[0] + self.assembler.full_dof[1]:]
            tip_prop, new_tip, split_face_list = analysis.remesh_at_tip(self.gb, pref, tref, self.fracture, self.min_face, newfrac, self.GAP)   
           
            pp.contact_conditions.set_projections(self.gb)
            g2d = self.gb.grids_of_dimension(2)[0]
            disp1, pres1 = analysis.mapping_solution(g2d, self.p, self.t, tips0, disp_cells, pres_cells, self.GAP)
    
            # disp0, pres0 = analysis.mapping_solution(g2d, self.p, self.t, tips0, pre_disp_cells, pre_pres_cells, self.GAP)
            self.gb.node_props(g2d)[pp.STATE]["u"] = disp1.reshape((g2d.num_cells*2,1))[:,0]
            self.gb.node_props(g2d)[pp.STATE]["p"] = pres1
            self.assembler.full_dof[0] = g2d.num_cells*2
            self.assembler.full_dof[1] = g2d.num_cells
    
            solution = np.concatenate((disp1.reshape((g2d.num_cells*2,1))[:,0], pres1, solution_1d))
            self.assembler.distribute_variable(solution)
            self.convergence_status = True
            # g2d.compute_geometry()
            self.update_discretize()
            
            g1d = self.gb.grids_of_dimension(1)
            
            dic_split = {}
            tip_prop_g1d = []
            new_tip_g1d = []
            for i in range(len(g1d)):
                nodes = g1d[i].nodes[[0,1],:].T
                split_face = np.array([], dtype = np.int32)
                
                
                for j in range(tip_prop.shape[0]):
                    dis = np.sqrt( (nodes[:,0] - tip_prop[j,0])**2 + (nodes[:,1] - tip_prop[j,1])**2 )
                    if min(dis) < np.finfo(float).eps*1E5:
                        tip_prop_g1d.append(tip_prop[j,:].reshape(1,2))
                        new_tip_g1d.append(new_tip[j,:].reshape(1,2))
                        split_face = np.append(split_face, np.int32(split_face_list[j]) )
    
                dic_split.update( { g1d[i]: split_face } )
                
            pp.propagate_fracture.propagate_fractures(self.gb, dic_split)
            
            for g, d in self.gb:
                if g.dim < self.Nd - 1:
                    # Should be really careful in this situation. Fingers crossed.
                    continue
    
                # Transfer information on new faces and cells from the format used
                # by self.evaluate_propagation to the format needed for update of
                # discretizations (see Discretization.update_discretization()).
                # TODO: This needs more documentation.
                new_faces = d.get("new_faces", np.array([], dtype=np.int32))
                split_faces = d.get("split_faces", np.array([], dtype=np.int32))
                modified_faces = np.hstack((new_faces, split_faces))
                update_info = {
                    "map_cells": d["cell_index_map"],
                    "map_faces": d["face_index_map"],
                    "modified_cells": d.get("new_cells", np.array([], dtype=np.int32)),
                    "modified_faces": d.get("new_faces", modified_faces),
                }
                d["update_discretization"] = update_info
    
            # Map variables after fracture propagation. Also initialize variables
            # for newly formed cells, faces and nodes.
            # Also map darcy fluxes and time-dependent boundary values (advection
            # and the div_u term in poro-elasticity).
            solution = self._map_variables(solution)
    
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
            
            frac_aft = []            
            for j, fracturej in enumerate(self.fracture):
                for k, tip_propi in enumerate(tip_prop_g1d):
                    if np.sum((fracturej[0,:] - tip_propi)**2) < np.finfo(float).eps*1E5:
                        fracturej = np.concatenate(( new_tip_g1d[k], fracturej ), axis = 0)
                    if np.sum((fracturej[-1,:] - tip_propi)**2) < np.finfo(float).eps*1E5:
                        fracturej = np.concatenate(( fracturej, new_tip_g1d[k] ), axis = 0)
        
                frac_aft.append(fracturej)
    
            self.fracture = frac_aft
            
            # super().after_newton_convergence(solution, errors, iteration_counter)
            self.assembler.distribute_variable(solution)
            self.convergence_status = True
            self.save_mechanical_bc_values()
            
            self.tips, frac_pts, frac_edges = analysis.fracture_infor(self.fracture)
            self.p, self.t = analysis.adjustmesh(g2d, self.tips, self.GAP)
            g2d = self.gb.grids_of_dimension(2)[0]
            data = self.gb.node_props(g2d)
            disp_cells = data[pp.STATE]["u"]
            pres_cells = data[pp.STATE]["p"]
            disp_nodes =  analysis.NN_recovery( disp_cells.reshape((self.t.shape[0],2)), self.p, self.t)
            pres_nodes =  analysis.NN_recovery( pres_cells, self.p, self.t)
            
        self.disp.append(disp_nodes)
        self.pres.append(pres_nodes)
        
        self.stored_disp.append(disp_cells)
        self.stored_pres.append(pres_cells)
        
        self.nodcoo.append(self.p)
        self.celind.append(self.t)
        self.facind.append(g2d.face_nodes.indices.reshape((2, g2d.num_faces), order='f').T   )
        self.farcoo.append(self.fracture)
        
        self.glotim.append(self.time)
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
                    
    def export_results(self):
        """
        Save displacement jumps and number of iterations for visualisation purposes. 
        These are written to file and plotted against time in Figure 4.
        """     
        g = self.gb.grids_of_dimension(2)[0]
        self.p, self.t = analysis.adjustmesh(g, self.tips, self.GAP)
        data = self.gb.node_props(g)
        disp = data[pp.STATE]["u"]
        pres = data[pp.STATE]["p"]
        
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.mechanics_parameter_key]
        
        traction = matrix_dictionary['stress'] * disp + matrix_dictionary["grad_p"] * pres

        return disp, pres, traction
    
        
mesh_size = 0.08
mesh_args = { "mesh_size_frac": mesh_size, "mesh_size_min": 1 * mesh_size, "mesh_size_bound": 5 * mesh_size } 
params = {"folder_name": "biot_2","convergence_tol": 2e-7,"max_iterations": 20,"file_name": "main_run",}


box = {"xmin": 0, "ymin": 0, "xmax": 4, "ymax": 4}  
fracture1 = np.array([[1.8, 2.0],[2.2, 2.0]])   
fracture = np.array([fracture1])

tips, frac_pts, frac_edges = analysis.fracture_infor(fracture)

setup = ModelSetup(box, fracture, mesh_args, params)
setup.prepare_simulation()
solver = pp.NewtonSolver(params)
k = 0
while setup.time < setup.end_time:
    setup.time += setup.time_step
    print('step = ', k, ': time = ', setup.time)
    solver.solve(setup)
    k+= 1
    
k = 36
disp_nodes = setup.disp[k]
pres_nodes = setup.pres[k]
p = setup.nodcoo[k]
t = setup.celind[k]
face = setup.facind[k]
# frac = np.concatenate((setup.farcoo[k][0], setup.farcoo[k][1]), axis = 0)
frac = setup.farcoo[k][0]
analysis.trisurf( p + disp_nodes*3e-1*0, t, fn = None, point = None, value = setup.stored_pres[k], infor = None)
analysis.trisurf( p + disp_nodes*3e-1*0, t, fn = None, point = None, value = setup.stored_disp[k][1::2], infor = None)
