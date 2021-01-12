from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import porepy as pp
import re_meshing

def trisurf(p, t, face = None, index = None):
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
        
    plt.show()
# trisurf(p, t, face = None, index = 1 ) 
mesh_size = 0.05
mesh_args = { "mesh_size_frac": mesh_size, "mesh_size_min": 1 * mesh_size, "mesh_size_bound": 3 * mesh_size } 
box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
fracture1 = np.array([[0.4, 0.6], [0.5, 0.65], [0.6, 0.6]])
fracture2 = np.array([[0.45, 0.2], [0.55, 0.2]])
fracture = np.array([fracture1, fracture2])

# newfrac = np.array([[0.55, 0.2], [0.59, 0.25]])
# newfrac = np.array([[0.45, 0.2], [0.41, 0.21]])
newfrac = np.array([[0.6, 0.6], [0.63, 0.65]])
# newfrac = np.array([[0.4, 0.6], [0.37, 0.65]])


tips, frac_pts, frac_edges = re_meshing.fracture_infor(fracture)
network = pp.FractureNetwork2d(frac_pts.T, frac_edges.T, domain=box)
gb = network.mesh(mesh_args)       
pp.contact_conditions.set_projections(gb)



g_2d = gb.grids_of_dimension(2)[0]
g_1d = gb.grids_of_dimension(1)
               

fno = g_2d.face_nodes.indices.reshape((g_2d.num_faces,2), order='c')
tips, frac_pts, frac_edges = re_meshing.fracture_infor(fracture)
p, t = re_meshing.adjustmesh(g_2d, tips)
trisurf(p, t, face = fno, index = 1 ) 

''' Remesh package'''
dic_split = re_meshing.remesh_at_tip(gb, fracture, newfrac)
# checking
g_2d = gb.grids_of_dimension(2)[0]
fn = g_2d.face_nodes.indices.reshape((g_2d.num_faces,2), order='c')
p, t = re_meshing.adjustmesh(g_2d, tips)
trisurf(p, t, face = fn, index = 1 ) 
''' splitting a face'''
pp.propagate_fracture.propagate_fractures(gb, dic_split)
''' checkout '''
g2d = gb.grids_of_dimension(2)[0]
p, t = re_meshing.adjustmesh(g2d, tips)
fn = g2d.face_nodes.indices.reshape((g2d.num_faces,2), order='c')
trisurf(p, t, fn) 

