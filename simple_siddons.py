import numpy as np

def siddons(src, trg):
    '''
    An implementation of Siddons raytracing, for a simple case.
    Assume 2D matrix, src & trg are outside of matrix.
    Things can get weird if the line from src -> trg does not intersect matrix...
    
    Inputs:
        src : 
            [x1,y1], coordinates of the ray start point.
        trg :
            [x2,y2], coordinates of the ray end point.
        N :
            number of pixels in the matrix. (assume square)
        dR :
            size of pixels in the matrix. (assume isotropic)
    '''
    eps = 1e-12 # for checking floating points against zero.
    
    # take x, y of src and trg
    x1, y1 = src
    x2, y2 = trg
    
    # number of planes in x and y directions
    Nx = N+1
    Ny = N+1
    
    # distance traversed over the x and y directions
    dx = x2 - x1 
    dy = y2 - y1

    #check for 0 distances
    if np.abs(dx) < eps:
        dx = eps
    if np.abs(dy) < eps:
        dy = eps

    coord_plane_1 = -N*dR//2

    def coord_plane(i):              # for i from 1 to N
        return coord_plane_1 + (i-1)*dR

    def aX(i):
        return (coord_plane(i) - x1) / dx

    def aY(i):
        return (coord_plane(i) - y1) / dy

    # parametric coords for intersections of ray with matrix
    a_min = max(min(aX(1), aX(Nx)), min(aY(1), aY(Ny)))
    a_max = min(max(aX(1), aX(Nx)), max(aY(1), aY(Ny)))

    # start and end index for i,j
    if dx >= 0:  # moving right
        i_min = Nx - (coord_plane(Nx) - a_min*dx - x1)/dR
        i_max = 1 + (x1 + a_max*dx - coord_plane_1)/dR
    else:
        i_min = Nx - (coord_plane(Nx) - a_max*dx - x1)/dR
        i_max = 1 + (x1 + a_min*dx - coord_plane_1)/dR

    if dy >= 0:  # moving up
        j_min = Ny - (coord_plane(Ny) - a_min*dy - y1)/dR
        j_max = 1 + (y1 + a_max*dy - coord_plane_1)/dR
    else:
        j_min = Ny - (coord_plane(Ny) - a_max*dy - y1)/dR
        j_max = 1 + (y1 + a_min*dy - coord_plane_1)/dR

    # round up/down for min/max, if not integer.
    if np.abs(j_min - np.round(j_min)) > eps:
        j_min = np.ceil(j_min)
    if np.abs(j_max - np.round(j_max)) > eps:
        j_max = np.floor(j_max)

    if np.abs(i_min - np.round(i_min)) > eps:
        i_min = np.ceil(i_min)
    if np.abs(i_max - np.round(i_max)) > eps:
        i_max = np.floor(i_max)

    # cast as integers
    i_min = np.round(i_min).astype(int)
    i_max = np.round(i_max).astype(int)
    j_min = np.round(j_min).astype(int)
    j_max = np.round(j_max).astype(int)

    i_vals = list(range(i_min, i_max+1, 1))
    j_vals = list(range(j_min, j_max+1, 1))

    # arrange a_x, a_y in ascending order
    if dx >= 0: 
        a_x = [aX(i) for i in i_vals]
    else:
        a_x = [aX(i) for i in i_vals[::-1]]

    if dy >= 0: 
        a_y = [aY(j) for j in j_vals]
    else:
        a_y = [aY(j) for j in j_vals[::-1]]

    # merge a_x, a_y into sorted alphas
    i = 0
    j = 0
    alphas = np.zeros(len(a_x) + len(a_y))
    while i<len(a_x) or j<len(a_y):
        if i<len(a_x) and j<len(a_y):
            if a_x[i] < a_y[j]:
                alphas[i+j] = a_x[i]
                i+=1
            else:
                alphas[i+j] = a_y[j]
                j+=1
        elif i<len(a_x):
            alphas[i+j] = a_x[i]
            i+=1
        elif j<len(a_y):
            alphas[i+j] = a_y[j]
            j+=1

    # get the difference between alphas (normalized lengths)
    dalphas = alphas[1:] - alphas[:-1]
    dST = src - trg
    d12 = np.linalg.norm(dST, axis=-1)
    l = dalphas * d12

    # get the voxel indices [i, j]
    Nl = len(dalphas)
    il = np.zeros(Nl, dtype=int)
    jl = np.zeros(Nl, dtype=int)
    m = 0
    for m in range(0, Nl):
        il[m] = (x1 + 0.5*(alphas[m]+alphas[m+1])*dx - coord_plane_1)/dR
        jl[m] = (y1 + 0.5*(alphas[m]+alphas[m+1])*dy - coord_plane_1)/dR

    return np.array([l, il, jl])