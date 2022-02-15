from IPython import get_ipython
get_ipython().magic('reset -sf') 
import time
import numpy as np
from typing import Dict
import scipy.sparse as sps
import scipy.interpolate as inter 
import porepy as pp
import mixedmode_fracture_analysis as analysis
# start = time.time()
class ModelSetup(pp.ContactMechanicsBiot, pp.ConformingFracturePropagation):
    def __init__(self, params):
        """ Set arguments for mesh size (as explained in other tutorials)
        and the name fo the export folder.
        """
        super().__init__(params)   
        self.length_scale = 1
        self.micro_size = 2.0*self.length_scale #0.23 0.33 0.5
        self.esp_p = 0.5
        self.esp_m = 1
        
        # ok
        nn = 9
        self.mesh_size = 0.005*self.length_scale*nn
        
        self.min_face = self.mesh_size/1
        self.fra_increment = self.mesh_size/self.esp_m
        self.mesh_size_micro = self.mesh_size/self.esp_m
        self.mesh_args = { "mesh_size_frac": 3*self.mesh_size/3, "mesh_size_min": 1 * self.mesh_size, "mesh_size_bound": 3* self.mesh_size} 
        

        self.box = {"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 2}  
        xcen =  (self.box['xmin'] + self.box['xmax'])/2
        ycen =  (self.box['ymin'] + self.box['ymax'])/2
        lenfra = 0.1*self.length_scale
        self.length_initial_fracture = lenfra
        
        phi = np.pi/4
        fracture1 = np.array([[xcen - lenfra/2*np.cos(phi), ycen - lenfra/2*np.sin(phi)],
                              [xcen + lenfra/2*np.cos(phi), ycen + lenfra/2*np.sin(phi)]]) 


        self.fracture = np.array([fracture1])

        self.initial_fracture = self.fracture.copy()
        self.GAP = 1.0001e-3*self.length_scale
 
        self.tips, self.frac_pts, self.frac_edges = analysis.fracture_infor(self.fracture)
        self.initial_aperture = 1.0001e-3*self.length_scale
        
        # Time seconds       
        self.end_time = 1000*60*60
        self.inject_time = [0*60*60, 1000*60*60]
        self.time_step = 1.0*60*60
        self.time_step_fix = 1.0*60*60
                
        self.glotim = []

        # solution 
        self.disp = []
        self.pres = []
        self.pres_1d = []
        self.cell_1d = []
        self.aper = []
        self.slip = []
        self.traction = []
        self.stored_aper = []
        self.stored_disp = []
        self.stored_pres = []
        self.traction = []
        # geometry
        self.nodcoo = []
        self.celind = []
        self.facind = []
        self.farcoo = []
        # Propagation criterion
        self.inj_cri = True
        
        self.gloinj = []
        self.lenfra = []
        self.glokeq = []
        self.glosif = []
        
        self.PROPAGATION = True
        self.MULTI_SCALE = False
        self.QPE = True
        

    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to granite and air.
        """
        self.YOUNG = 40e9
        self.POISSON = 0.2
        self.BIOT = 0.8
        self.POROSITY = 1E-2
        self.COMPRESSIBILITY = 4.4E-10
        self.PERMEABILITY = 5e-20
        self.VISCOSITY = 1.0E-4
        self.FRICTION = 0.5
        self.FLUID_DENSITY = 930*0
        self.ROCK_DENSITY = 2.7E3*0
       
        self.INJECTION = 5e-9 #( m2/s)
        self.SH = 20E6#/self.box["ymax"]
        self.Sh = 10E6#/self.box["xmax"]

        self.BULK = self.YOUNG/3/(1 - 2*self.POISSON)
        self.LAMBDA = self.YOUNG*self.POISSON / ((1 + self.POISSON) * (1 - 2 * self.POISSON))
        self.MU = self.YOUNG/ (2 * (1 + self.POISSON))
        self.material = dict([('YOUNG', self.YOUNG), ('POISSON', self.POISSON), ('KIC', 0.7e6) ])      
    def porosity(self, g) -> float:
        if g.dim == 2:
            return self.POROSITY
        else:
            return 0.1
    def biot_alpha(self, g) -> np.ndarray:
        if g.dim == 2:
            return self.BIOT
        else:
            return 1.0
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
        self.min_cell = np.min(g2d.cell_volumes)
        self.p, self.t = analysis.adjustmesh(g2d, self.tips, self.GAP)
        self.fa_no =  g2d.face_nodes.indices.reshape((2, g2d.num_faces), order='f').T 
        self.pc = g2d.cell_centers[[0, 1], :].T
        self.update_all_apertures()
        return gb

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
    def before_newton_iteration(self) -> None:
        self._iteration += 1
        self.update_all_apertures(to_iterate=True)
        self.set_parameters()
        term_list = [
            "!mpsa",
            "!stabilization",
            "!div_u",
            "!grad_p",
            "!diffusion",
        ]
        filt = pp.assembler_filters.ListFilter(term_list=term_list)
        self.assembler.discretize(filt=filt)

        for dim in range(self.Nd):
            grid_list = self.gb.grids_of_dimension(dim)
            if len(grid_list) == 0:
                continue
            filt = pp.assembler_filters.ListFilter(
                grid_list=grid_list,
                term_list=["diffusion"],
            )
            self.assembler.discretize(filt=filt)
        
    def aperture(self, g, from_iterate=True) -> np.ndarray:
        """
        Obtain the aperture of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE][pp.ITERATE]["aperture"]
        else:
            return self.gb.node_props(g)[pp.STATE]['iterate']['aperture']  
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
                lam = self.LAMBDA * np.ones(g.num_cells)
                mu = self.MU * np.ones(g.num_cells)
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
                    {"friction_coefficient": friction, 
                     "time_step": self.time_step,
                     "time": self.time,
                     "dilation_angle": np.radians(1),
                     },
                )

        for _, d in gb.edges():
            mg: pp.MortarGrid = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    def set_scalar_parameters(self):
        for g, d in self.gb:
            bc = self._bc_type_scalar(g)
            bc_values = self._bc_values_scalar(g)
            source_values = self.source_scalar(g)
            specific_volume = self.specific_volume(g)
            alpha = self.biot_alpha(g)
            if g.dim == 2:
                mass_weight = self.COMPRESSIBILITY * self.porosity(g) + (self.BIOT - self.porosity(g)) / self.BULK * specific_volume
            else:
                mass_weight = self.COMPRESSIBILITY * self.porosity(g) * specific_volume           
            
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
            # We assume isotropic permeability in the fracture, i.e. the normal permeability equals the tangential one
            k_s = data_l[pp.PARAMETERS][key]["second_order_tensor"].values[0, 0]
            # Division through half the aperture represents taking the (normal) gradient
            kn = mg.secondary_to_mortar_int() * np.divide(k_s, a * V / 2)
            tr = np.abs(g_h.cell_faces)
            V_j = mg.primary_to_mortar_avg() * tr * V_h
            kn = kn * V_j
            pp.initialize_data(mg, d, key, {"normal_diffusivity": kn})
    def _source_mechanics(self, g):
        Fm = np.zeros(g.num_cells * self._Nd) 
        Fm[1::2] = self.ROCK_DENSITY*g.cell_volumes

        return Fm

    def source_scalar(self, g):
        values = np.zeros(g.num_cells)
        # if g.dim == 1 and self.inj_cri: #(self.time >= 20*60 and self.time <= 90*60) and 
        if g.dim == 1 and self.time >= self.inject_time[0] and self.time <= self.inject_time[1]: 
        # if g.dim == 1 and self.time >= 50*60:
            print('injection')
            frac1 = self.initial_fracture[0]
            poix = (g.cell_centers[0,:] <= np.max(frac1[:,0])) * (g.cell_centers[0,:] >= np.min(frac1[:,0]))
            poiy = (g.cell_centers[1,:] <= np.max(frac1[:,1]) + self.initial_aperture) * (g.cell_centers[1,:] >= np.min(frac1[:,1]) - self.initial_aperture)
            inject =  poix*poiy
            qeq = self.INJECTION/self.length_initial_fracture/self.initial_aperture #(1/s)
            values[inject] = qeq*self.time_step*g.cell_volumes[inject]*self.initial_aperture #(inter)
        return values

    
    def _bc_type_mechanics(self, g):
        # dir = displacement 
        # neu = traction
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g)

        bc.is_dir[0, west] = True
        bc.is_dir[1, south] = True
        
        frac1 = self.initial_fracture[0]
        poix = (g.face_centers[0,:] <= np.max(frac1[:,0])) * (g.face_centers[0,:] >= np.min(frac1[:,0]))
        poiy = (g.face_centers[1,:] <= np.max(frac1[:,1]) + self.initial_aperture) * (g.face_centers[1,:] >= np.min(frac1[:,1]) - self.initial_aperture)
        inject =  poix*poiy
        
        frac_face = g.tags["fracture_faces"]#*inject
        bc.is_dir[:, frac_face] = True
        # print( np.where(frac_face))
        
        return bc
    def _bc_values_mechanics(self, g):
        # Set the boundary values
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))

        values[1, north] = -self.Sh*g.face_areas[north]
        # values[1, south] = self.Sh*g.face_areas[south]
        values[0, east] = -self.SH*g.face_areas[east]
        # values[0, west] = self.SH*g.face_areas[east]

        return values.ravel("F")
    def _bc_type_scalar(self, g):
        bc = pp.BoundaryCondition(g)
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc.is_neu[east] = True
        bc.is_neu[west] = True
        bc.is_neu[north] = True
        bc.is_neu[south] = True
        frac_face = g.tags["fracture_faces"]#*inject
        bc.is_neu[frac_face] = True
        return bc

    def _bc_values_scalar(self, g):
        bc_values = np.zeros(g.num_faces)
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        return bc_values
    
    def initial_condition(self):
        """
        Initial guess for Newton iteration, scalar variable and bc_values (for time
        discretization).
        """
        super().initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            initial_scalar_value = np.zeros(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})
            if g.dim == self.Nd:
                bc_values = self.bc_values_mechanics(g)
                mech_dict = {"bc_values": bc_values}
                d[pp.STATE].update({self.mechanics_parameter_key: mech_dict})

        for _, d in self.gb.edges():
            mg = d["mortar_grid"]
            initial_value = np.zeros(mg.num_cells)
            d[pp.STATE][self.mortar_scalar_variable] = initial_value
            
    def after_newton_convergence(self, solution, errors, iteration_counter):
        """Propagate fractures if relevant. Update variables and parameters
        according to the newly calculated solution.
        
        """
        
        self.pro_cri = False
        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        self.save_mechanical_bc_values()

        g2d = self.gb.grids_of_dimension(2)[0]
        data = self.gb.node_props(g2d)
        
        g1d = self.gb.grids_of_dimension(1)[0]
        data1 = self.gb.node_props(g1d)
        trac = data1[pp.STATE]["contact_traction"].reshape((g1d.num_cells,2))
        nod_trac = g1d.cell_centers[[0, 1],:].T
        # We export the converged solution *before* propagation:
        apertures, norm_u_tau = self.update_all_apertures(to_iterate=True)
        self.slip.append(norm_u_tau)
        self.export_step()
        
        apertures = self.aperture(g1d, from_iterate = True)
        
        disp_cells = data[pp.STATE]["u"].reshape((self.t.shape[0],2))
        pres_cells = data[pp.STATE]["p"]
        
        # disp_nodes =  analysis.NN_recovery( disp_cells, self.p, self.t)
        # analysis.trisurf( self.p + disp_nodes*5e2, self.t) 

        
        # Store data/results
        self.stored_aper.append(apertures)
        self.stored_disp.append(disp_cells)
        self.stored_pres.append(pres_cells)
        self.nodcoo.append(self.p)
        self.celind.append(self.t)
        self.facind.append(g2d.face_nodes.indices.reshape((2, g2d.num_faces), order='f').T   )
        self.farcoo.append(self.fracture)
        
        self.pres_1d.append(data1[pp.STATE]["p"])
        self.cell_1d.append(nod_trac)
        self.aper.append(apertures)
        self.traction.append(np.concatenate((nod_trac,trac), axis = 1))
        
        
        tips, frac_pts, frac_edges = analysis.fracture_infor(self.fracture)
        
            
        if len(self.fracture) == 2:
            pt1 = self.fracture[0][0]; pt2 = self.fracture[0][-1]
            segments = self.fracture[-1]
            intpoi = analysis.intersectLines( pt1, pt2, segments )
            if len(intpoi) > 0:
                dis = np.sqrt((tips[:,0] - intpoi[0,0])**2 + (tips[:,1] - intpoi[0,1])**2)
                index = np.where(dis <= np.finfo(float).eps*1E5)[0]
                tips_actualy = np.delete(tips, index, axis = 0)
            else:
                tips_actualy = tips
        else:
            tips_actualy = tips
         
        pmod, tmod = analysis.adjustmesh(g2d, tips, self.GAP) 
        
        # starttime = time.time()
        if self.MULTI_SCALE:
            fracture_small = self.fracture.copy()
            tips_small = tips.copy()
            pro_step = 0
            newfrac = []
            for i in range(tips_small.shape[0]):
                newfrac.append( np.array([tips[i], tips[i]]) )
            while pro_step <= self.esp_m - 1:
                keq, ki, newfrac, fracture_small = self.propagation_small_scale( fracture_small, newfrac, tips, disp_cells, pres_cells, self.QPE )
                print(keq)
                pro_step = pro_step + 1
                if np.max(np.abs(keq)) < self.material['KIC']:
                    break
            kk = 0
            tip_update = np.copy(tips)
            for j, fracturej in enumerate(fracture_small):
                tip_update[2*kk,:] = fracturej[0,:]
                tip_update[2*kk + 1,:] = fracturej[-1,:]
                kk = kk+1
        else:       
            pref, tref = analysis.refinement( pmod, tmod, self.p, self.t, self.fracture, tips_actualy, self.min_cell, self.min_face, self.GAP)
            # sol1 = analysis.NN_recovery( disp_cells, self.p, self.t)                    
            # sol2 = analysis.linear_interpolation(self.p,self.t, sol1, pref)
            # valnod = inter.griddata(self.pc, disp_cells, pref, method = 'cubic')
            # valnod[frac_nod,:] = sol2[frac_nod,:]
            
            keq, ki, newfrac, tip_update, p6, t6, disp6 = analysis.evaluate_propagation(self.material, pref, tref, self.p, self.t, 
                                                                                   self.initial_fracture, self.fracture, tips_actualy, 
                                                                                   self.min_face, disp_cells, pres_cells, self.GAP, self.QPE)
            print(keq)
        
        # endtime = time.time()   
        # print('computational time', endtime - starttime)

         
        self.glosif.append(ki)
        self.glokeq.append(keq)
        
        if self.PROPAGATION:
            newfrac0 = []
            for i in range(len(newfrac)):
                lfi = np.sqrt(np.sum( (newfrac[i][1,:] - newfrac[i][0,:])**2 ))
                if lfi >= self.min_face*self.esp_p:
                    newfrac0.append(newfrac[i])
                    
            newfrac = newfrac0.copy()   
            if len(newfrac) > 0:
                dis = []
                for i in range(len(newfrac)):
                    tipinew = newfrac[i][1,:]
                    disi = np.min(np.array([np.abs(tipinew[0] - self.box['xmin']),
                                            np.abs(tipinew[0] - self.box['xmax']),
                                            np.abs(tipinew[1] - self.box['ymin']),
                                            np.abs(tipinew[1] - self.box['ymax'])]))
                    dis.append(disi)
                if np.min(dis) > self.min_face*5:                   
                    self.pro_cri = True
        
                    solution_1d = solution[self.assembler.full_dof[0] + self.assembler.full_dof[1]:]
                    if self.MULTI_SCALE:
                        lenfac = np.abs(newfrac[0][1,0] - newfrac[0][0,0])
                        pref, tref = analysis.refinement( pmod, tmod, self.p, self.t, self.fracture, tips, self.min_cell*1/3, self.min_face*2/3, self.GAP)
                
                    tip_prop, new_tip, split_face_list = analysis.remesh_at_tip(self.gb, pref, tref, self.fracture, self.min_face, newfrac, self.GAP)   
                    
                    pp.contact_conditions.set_projections(self.gb)
                    g2d = self.gb.grids_of_dimension(2)[0]
                
                    disp1, pres1 = analysis.mapping_solution(g2d, self.p, self.t, tip_update, disp_cells, pres_cells, self.GAP)
            
                    # disp0, pres0 = analysis.mapping_solution(g2d, self.p, self.t, tips0, pre_disp_cells, pre_pres_cells, self.GAP)
                    self.gb.node_props(g2d)[pp.STATE]["u"] = disp1.reshape((g2d.num_cells*2,1))[:,0]
                    self.gb.node_props(g2d)[pp.STATE]["p"] = pres1
                    self.assembler.full_dof[0] = g2d.num_cells*2
                    self.assembler.full_dof[1] = g2d.num_cells
            
                    solution = np.concatenate((disp1.reshape((g2d.num_cells*2,1))[:,0], pres1, solution_1d))
        
                    self.assembler.distribute_variable(solution)
                    self.convergence_status = True            
                    
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
        
                    self.update_discretize()
                    # For now, update discretizations will do a full rediscretization
                    # TODO: Replace this with a targeted rediscretization.
                    # We may want to use some of the code below (after return), but not all of
                    # it.
                    # self._minimal_update_discretization()
                    super().after_newton_convergence(solution, errors, iteration_counter)
                    
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
                    # self.assembler.distribute_variable(solution)
                    # self.convergence_status = True
                    # self.save_mechanical_bc_values()
                    
                    self.tips, frac_pts, frac_edges = analysis.fracture_infor(self.fracture)
                    self.p, self.t = analysis.adjustmesh(g2d, self.tips, self.GAP)
                    g2d = self.gb.grids_of_dimension(2)[0]
                    self.pc = g2d.cell_centers[[0, 1], :].T
                    data = self.gb.node_props(g2d)
                    # disp_cells = data[pp.STATE]["u"].reshape((self.t.shape[0],2))
                    # pres_cells = data[pp.STATE]["p"]            
            
        if self.pro_cri:
            # lenfac = np.sqrt(np.sum((newfrac[0][1,:] - newfrac[0][0,:])**2))
            lenfac = np.abs(newfrac[0][1,0] - newfrac[0][0,0])
            self.lenfra.append(lenfac)
            self.inj_cri = False
        else:
            self.lenfra.append(0)
        if self.inj_cri:
            self.gloinj.append(self.INJECTION)
        else:
            self.gloinj.append(0)
            
        self.glotim.append(self.time)
        
        self.adjust_time_step()
        
        
        
        return
    def propagation_small_scale(self, fracture_small, newfrac, tips, disp_cells, pres_cells, QPE):    
        kk = 0
        tips_small = np.copy(tips)
        for j, fracturej in enumerate(fracture_small):
            tips_small[2*kk,:] = fracturej[0,:]
            tips_small[2*kk + 1,:] = fracturej[-1,:]
            kk = kk+1
             
        G = []; keq = []; ki = []; craang = []; iniang = [];     
        for i in range(len(tips_small)):
            tipi = tips_small[i] 
            box_small = {"xmin": tipi[0] - self.micro_size/2, 
                         "ymin": tipi[1] - self.micro_size/2, 
                         "xmax": tipi[0] + self.micro_size/2, 
                         "ymax": tipi[1] + self.micro_size/2}

            segments = np.array([[box_small["xmin"], box_small["ymin"]], 
                                 [box_small["xmax"], box_small["ymin"]],
                                 [box_small["xmax"], box_small["ymax"]],
                                 [box_small["xmin"], box_small["ymax"]],
                                 [box_small["xmin"], box_small["ymin"]]])
            
            for j, fracturej in enumerate(fracture_small):
                # tips_small = fracturej[[0, -1],:]
                frac_box = False
                if np.sum((fracturej[0,:] - tipi)**2) < np.finfo(float).eps*1E5:
                    frac_box = True
                    intpoi = []
                    for ii in range(len(fracturej) - 1):
                        intpoi_i = analysis.intersectLines( fracturej[ii,:], fracturej[ii + 1,:], segments )
                        if len(intpoi_i) > 0:
                            intpoi = intpoi_i #analysis.intersectLines( fracturej[0,:], fracturej[-1,:], segments )
                    if len(intpoi) == 0:
                        disl = np.abs(fracturej[-1,0] - box_small["xmin"])
                        disr = np.abs(fracturej[-1,0] - box_small["xmax"])
                        disb = np.abs(fracturej[-1,1] - box_small["ymin"])
                        dist = np.abs(fracturej[-1,1] - box_small["ymax"])
                        frac_small_modify = np.copy(fracturej)
                        if disl < self.fra_increment/2:
                            frac_small_modify[-1,0] = box_small["xmin"]
                        if disr < self.fra_increment/2:
                            frac_small_modify[-1,0] = box_small["xmax"]
                        if disb < self.fra_increment/2:
                            frac_small_modify[-1,1] = box_small["ymin"]
                        if dist < self.fra_increment/2:
                            frac_small_modify[-1,1] = box_small["xmax"]
                            
                        frac_small = np.array([frac_small_modify]) 
                    else:
                        r = np.sqrt( np.sum( (intpoi - tipi)**2 ) )
                        ri = np.sqrt( ( (fracturej[:,0] - tipi[0])**2 + (fracturej[:,1] - tipi[1])**2 ) )
                        insbox = np.where(ri < r)[0]  
                        dis = np.sqrt(np.sum((fracturej[insbox[-1],:] - intpoi[0])**2))
                        if dis < self.fra_increment/2:
                            frac_small = np.array( [np.concatenate( (fracturej[insbox[0:-1],:], intpoi), axis = 0)] )
                        else:
                            frac_small = np.array( [np.concatenate( (fracturej[insbox,:], intpoi), axis = 0)] )
                       
                if np.sum((fracturej[-1,:] - tipi)**2) < np.finfo(float).eps*1E5:
                    frac_box = True
                    intpoi = []
                    for ii in range(len(fracturej) - 1):
                        intpoi_i = analysis.intersectLines( fracturej[ii,:], fracturej[ii + 1,:], segments )
                        if len(intpoi_i) > 0:
                            intpoi = intpoi_i #analysis.intersectLines( fracturej[0,:], fracturej[-1,:], segments )
                    if len(intpoi) == 0:
                        disl = np.abs(fracturej[0,0] - box_small["xmin"])
                        disr = np.abs(fracturej[0,0] - box_small["xmax"])
                        disb = np.abs(fracturej[0,1] - box_small["ymin"])
                        dist = np.abs(fracturej[0,1] - box_small["ymax"])
                        frac_small_modify = np.copy(fracturej)
                        if disl < self.fra_increment/2:
                            frac_small_modify[0,0] = box_small["xmin"]
                        if disr < self.fra_increment/2:
                            frac_small_modify[0,0] = box_small["xmax"]
                        if disb < self.fra_increment/2:
                            frac_small_modify[0,1] = box_small["ymin"]
                        if dist < self.fra_increment/2:
                            frac_small_modify[0,1] = box_small["xmax"]
                            
                        frac_small = np.array([frac_small_modify]) 
                    else:
                        r = np.sqrt( np.sum( (intpoi - tipi)**2 ) )
                        ri = np.sqrt( ( (fracturej[:,0] - tipi[0])**2 + (fracturej[:,1] - tipi[1])**2 ) )
                        insbox = np.where(ri < r)[0]   
                        dis = np.sqrt(np.sum((fracturej[insbox[0],:] - intpoi[0])**2))
                        if dis < self.fra_increment/2:
                            frac_small = np.array( [np.concatenate( (intpoi, fracturej[insbox[1::],:]), axis = 0)] )
                        else:
                            frac_small = np.array( [np.concatenate( (intpoi, fracturej[insbox,:]), axis = 0)] )
                        
            if frac_box:        
                tips_small0, frac_pts_small, frac_edges_small = analysis.fracture_infor(frac_small)
                mesh_args = { "mesh_size_frac": self.mesh_size_micro, "mesh_size_min": 2 * self.mesh_size_micro, "mesh_size_bound": 12 * self.mesh_size_micro}
                network_small = pp.FractureNetwork2d( frac_pts_small.T, frac_edges_small.T, domain=box_small )
                gb_small = network_small.mesh(mesh_args)   
                pp.contact_conditions.set_projections(gb_small)
                g2d_small = gb_small.grids_of_dimension(2)[0]  
                p0 = g2d_small.nodes
                t0 = g2d_small.cell_nodes().indices.reshape((3, g2d_small.num_cells), order='f').T
                p0 = p0[[0,1],:].T
                dis2tip = np.sqrt((p0[:,0] - tipi[0])**2 + (p0[:,1] - tipi[1])**2)
                tipind = np.where(dis2tip == np.min(dis2tip))[0] 
                
                face_ind = np.reshape(g2d_small.face_nodes.indices, (g2d_small.dim, -1), order='F').T
                frac_fac = np.where(g2d_small.tags['fracture_faces'])[0]  
                frac_nod = np.where(g2d_small.tags['fracture_nodes'])[0]  

                node_adj_old = []
                for _, frac_faci in enumerate(frac_fac):
                    index = face_ind[frac_faci,:]
                    ele = np.intersect1d(np.where(t0 == index[0])[0], np.where(t0 == index[1])[0])
                    index0 = np.setdiff1d(t0[ele,:], index)[0]
                    A, B = p0[index,:]
                    M = p0[index0,:]
                    N,_,_= analysis.projection(A,B,M)
                    normi = (N - M); normi = normi/np.sqrt(sum(normi**2))
                    node_adj = np.setdiff1d(np.setdiff1d(index,tipind),node_adj_old)
                    p0[node_adj,:] = p0[node_adj,:] - normi*1E-3/2
                    node_adj_old = np.concatenate((node_adj_old,node_adj))
                p_small, t_small = p0, t0
                
                sol1 = analysis.NN_recovery( disp_cells, self.p, self.t)                    
                dispnod = analysis.linear_interpolation(self.p,self.t, sol1, p_small)
                
                sol2 = analysis.NN_recovery( pres_cells, self.p, self.t)                    
                presnod = analysis.linear_interpolation(self.p,self.t, sol2, p_small)

                # p_small, t_small = analysis.adjustmesh(g2d_small, tips_small, 1E-3)  

                Gi, sifi, keqi, craangi, iniangi = analysis.evaluate_propagation_small(self.material, p_small, t_small, frac_small, tipi, 
                                                                                   self.p, self.t, dispnod, presnod, self.initial_fracture, self.fra_increment, 1E-3, QPE)
                G = np.append(G, Gi); 
                keq = np.append(keq, keqi); 
                ki.append(sifi[0]); 
                craang = np.append(craang, craangi); 
                iniang = np.append(iniang, iniangi)        
                    
        ladv = np.zeros(tips_small.shape[0])
        pos_pro = []
        if np.max(np.abs(keq)) >= self.material['KIC']:
            pos_pro = np.where(np.abs(keq)*1.1 >= self.material['KIC'])[0]
        
        if len(pos_pro) > 0:
            ladv[pos_pro] = self.fra_increment#*( G[pos_pro]/np.max(G[pos_pro]) )**0.35
            
        
        tip_initial = np.copy(tips_small)
        tip_new = np.copy(tips_small)
        for i in range(tips_small.shape[0]):
            if ladv[i] >= self.fra_increment*0.5:
                tipi = tips_small[i]
                tipnew = tipi + ladv[i]*np.array([np.cos(craang[i] + iniang[i]), np.sin(craang[i] + iniang[i])])
                newfrac[i][1,:] = tipnew
                tip_new[i,:] = tipnew
    
        
        for i in range(len(newfrac)):
            lfi = np.sqrt(np.sum( (newfrac[i][1,:] - newfrac[i][0,:])**2 ))
            if lfi >= self.fra_increment*0.5:
                tip0 = tip_initial[i,:]
                tip1 = tip_new[i,:]
                frac_aft = []            
                for j, fracturej in enumerate(fracture_small):
                    if np.sum((fracturej[0,:] - tip0)**2) < np.finfo(float).eps*1E6:
                        fracturej = np.concatenate(( tip1.reshape(1,2), fracturej ), axis = 0)
                    if np.sum((fracturej[-1,:] - tip0)**2) < np.finfo(float).eps*1E6:
                        fracturej = np.concatenate(( fracturej, tip1.reshape(1,2) ), axis = 0)
        
                    frac_aft.append(fracturej)   
                fracture_small = frac_aft
        
        return keq, ki, newfrac, fracture_small
    def update_all_apertures(self, to_iterate=True):
        """
        To better control the aperture computation, it is done for the entire gb by a
        single function call. This also allows us to ensure the fracture apertures
        are updated before the intersection apertures are inherited.
        The aperture of a fracture is
            initial aperture + || u_n ||
        """
        norm_u_tau = []
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
                    norm_u_n = np.absolute(u_mortar_local[-1])
                    norm_u_tau = np.linalg.norm(u_mortar_local[:-1], axis=0)
                    # Absolute value to avoid negative volumes for non-converged
                    # solution (if from_iterate is True above)
                    # apertures += np.absolute(u_mortar_local[-1])
                    dilation_angle = norm_u_n*0
                    dilation_angle[norm_u_n != 0] = np.arctan(norm_u_tau[norm_u_n != 0]/norm_u_n[norm_u_n != 0])
                    # apertures += ( u_mortar_local[-1] + np.cos(dilation_angle) * norm_u_tau )
                    apertures += norm_u_n

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

        return apertures, norm_u_tau
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
        
        
        
        self.set_parameters()
        
        gb = self.gb

        # t_0 = time.time()

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

        self.discretize_biot(update_after_geometry_change=True)
        print('True')

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

        # Build a list of all edges, and all couplings
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
        
        self.initialize_linear_solver()   
      
    def adjust_time_step(self):
        if self.pro_cri:
            self.time_step = self.time_step_fix/100
        else:
            self.time_step = self.time_step_fix
        
params = {"convergence_tol": 1e-7, "max_iterations": 50}
setup = ModelSetup(params)
setup.prepare_simulation()
solver = pp.NewtonSolver(params)

# p = setup.p
# t = setup.t
# face = setup.fa_no
# analysis.trisurf(  p, t, fn = face, point = None, value = None, vmin = None, vmax = None, infor = 1)

# analysis.trisurf(  p_small, t_small, fn = None, point = None, value = None, infor = None)


k = 0
while setup.time <= setup.end_time and setup.tips[0,0] >= 0.68:
    print('step = ', k, ': time = ', setup.time/60)
    # if setup.time/60 == 102.1:
    if k == 4:
        stop = 1
    solver.solve(setup)
    k+= 1
    setup.time += setup.time_step
    
# start = time.time()
# end = time.time()
# print(end - start)    
kk = k - 1
# kk = 126
trac = setup.traction
p = setup.nodcoo[kk]
t = setup.celind[kk]
face = setup.facind[kk]
disp_nodes =  analysis.NN_recovery( setup.stored_disp[kk], p, t)
pres_nodes =  analysis.NN_recovery( setup.stored_pres[kk], p, t)
# frac = np.concatenate((setup.farcoo[k][0], setup.farcoo[k][1]), axis = 0)
frac = setup.farcoo[kk][0]
analysis.trisurf( p + disp_nodes*1e2, t, fn = None, point = None, value = setup.stored_pres[kk]*1e-6, vmin = 6.5, vmax = 13)

disp = np.sqrt( setup.stored_disp[kk][:,0]**2 +  setup.stored_disp[kk][:,1]**2)
# disp = setup.stored_disp[k][:,1]
analysis.trisurf( p + disp_nodes*1e2, t, fn = None, point = None, value = disp, vmin = None, vmax = None, infor = None)

xx1 = setup.stored_pres[kk]
xx2 = setup.stored_disp[kk]
xx = np.concatenate( (xx1.reshape(len(xx1), 1),xx2), axis = 1)


# import pandas as pd
# xx1 = setup.cell_1d

# xx2 = setup.aper
# df = pd.DataFrame(xx2)
# df.to_csv('apertures.xlsx')

# xx3 = setup.slip
# df = pd.DataFrame(xx3)
# df.to_csv('slip.xlsx')

# xx4 = setup.pres_1d
# df = pd.DataFrame(xx4)
# df.to_csv('pressure.xlsx')

# xxx = []
# for i in range(len(xx1)):
#     xxx.append(xx1[i].reshape(len(xx1[i])*2, 1)[:,0] )
# df = pd.DataFrame(xxx)
# df.to_csv('coord.xlsx')


# xx4 = setup.traction
# xxx = []
# for i in range(len(xx4)):
#     xxx.append(xx4[i][:,[2,3]].reshape(len(xx4[i])*2, 1)[:,0] )
# df = pd.DataFrame(xxx)
# df.to_csv('traction.xlsx')

