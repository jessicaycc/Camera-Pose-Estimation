import numpy as np
from numpy.linalg import inv, lstsq
from scipy.ndimage import gaussian_filter

def saddle_point(I):
        """
        Locate saddle point in an image patch.

        The function identifies the subpixel centre of a cross-junction in the
        image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
        finding the critical point of that paraboloid.

        Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
        left corner of the patch, i.e., the pixels are treated as covering an 
        area of one unit square.

        Parameters:
        -----------
        I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

        Returns:
        --------
        pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
        """
        #--- FILL ME IN ---

        m, n = I.shape
        k = 0
        M = np.empty([m*n, 6])
        
        # Quadratic fitting with least square (Ax-b)^2. 
        # Contruct the b matrix 
        N = I.flatten()

        # construct the A matrix  where each row iis [x^2 xy y^2 x y 1] 
        # (x, y) represents pixel locations (denoting as i and j respectively here)
        for i in range (0, m):
                for j in range(0,n):
                        a = np.array([j**2, i*j, i**2, j, i, 1])
                        M[k,:] = a
                        k+=1
        
        # linear least square to find the coefficients 
        alpha, beta, gamma, delta, epsilon, zeta = lstsq(M, N, rcond = None)[0]
        
        # find the coordinates (to subpixal accuracy) by solving the equaiton after 
        # equation 4 in the Lucchese paper
        H = np.array([ [2 * alpha, beta], 
                       [beta, 2 * gamma]])

        P = np.array( [ [delta], 
                        [epsilon] ])

        pt = -np.matmul(inv(H), P) 
        #------------------

        return pt
