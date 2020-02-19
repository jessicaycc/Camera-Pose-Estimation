import numpy as np
from scipy.ndimage.filters import *
from scipy.linalg import null_space



def cross_junctions(I, bounds, Wpts):
        """
        Find cross-junctions in image with subpixel accuracy.

        The function locates a series of cross-junction points on a planar 
        calibration target, where the target is bounded in the image by the 
        specified quadrilateral. The number of cross-junctions identified 
        should be equal to the number of world points.

        Note also that the world and image points must be in *correspondence*,
        that is, the first world point should map to the first image point, etc.

        Parameters:
        -----------
        I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
        bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
        Wpts    - 3xn np.array of world points (in 3D, on calibration target).

        Returns:
        --------
        Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
                left Wpts of I. These should be floating-point values.
        """
        #--- FILL ME IN ---
        Ipts = np.zeros((2, 48))
        # Grid size in world frame 
        grid = Wpts[0,1] - Wpts[0,0]
        
        # Estimate the 4 corners of world points 
        WptsX_min = min(Wpts[0,:]) - grid * 3/2
        WptsX_max = max(Wpts[0,:]) + grid * 3/2
        WptsY_min = min(Wpts[1,:]) - grid * 5/4
        WptsY_max = max(Wpts[1,:]) + grid * 5/4

        # Create a bounding box 
        imagePlane = np.array([[WptsX_min, WptsX_max, WptsX_max, WptsX_min],
                               [WptsY_min, WptsY_min, WptsY_max, WptsY_max]])
       
        # Get homography matrix 
        H, A = dlt_homography(imagePlane, bounds)

        # Homography transform Wpts to the other frame
        Wpts[-1] = 1
        X = H @ Wpts
        X /= X[-1]
        X = np.round(X[:-1]).astype(int).T
        
        # Create a patch around each points and use saddle point to refine them 
        window = 20 
        for i in range (48):
                ymin = X[i,1]-window
                ymax = X[i,1]+window
                xmin = X[i,0]-window
                xmax = X[i,0]+window

                patch = I[ymin:ymax+1, xmin:xmax+1]

                pt = (saddle_point(patch)).flatten()
                pt = np.array(pt).T

                Ipts[:,i] = pt + X[i] - window

        #------------------

        return Ipts

def dlt_homography(I1pts, I2pts):
        """
        Find perspective Homography between two images.

        Given 4 points from 2 separate images, compute the perspective homography
        (warp) between these points using the DLT algorithm.

        Parameters:
        ----------- 
        I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
        I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

        Returns:
        --------
        H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
        A  - 8x9 np.array of DLT matrix used to determine homography.
        """
        #--- FILL ME IN ---

        first = True

        # Construct the A matrix 
        for (x, y, u, v) in zip(I1pts[0], I1pts[1], I2pts[0], I2pts[1]):
                M = np.array([[-x, -y, -1, 0, 0, 0, u * x, u * y, u], 
                             [0, 0, 0, -x, -y, -1, v * x, v * y, v]])
                if first:
                        A = np.copy(M)
                        first = False 
                else: 
                        A = np.append(A, M, axis = 0) 
        # Find the solution by finding the nullspace of A
        H = null_space(A)
        H = H[:, -1]
        H = H.reshape(3,3) 
        # Normalization
        H = H/H[-1,-1]
        #------------------

        return H, A

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
        N = I.flatten()
     
        for i in range (0, m):
                for j in range(0,n):
                        a = np.array([j**2, i*j, i**2, j, i, 1])
                        M[k,:] = a
                        k+=1
        
        alpha, beta, gamma, delta, epsilon, zeta = np.linalg.lstsq(M, N, rcond = None)[0]
        
        H = np.array([ [2 * alpha, beta], 
                       [beta, 2 * gamma]])

        P = np.array( [ [delta], 
                        [epsilon] ])

        pt = -np.matmul(np.linalg.inv(H), P) 
        #------------------

        return pt
