import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
        """
        Estimate camera pose from 2D-3D correspondences via NLS.

        The function performs a nonlinear least squares optimization procedure 
        to determine the best estimate of the camera pose in the calibration
        target frame, given 2D-3D point correspondences.

        Parameters:
        -----------
        Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
        Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
        Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

        Returns:
        --------
        Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
        """
        maxIters = 250                          # Set maximum iterations.

        tp = Ipts.shape[1]                      # Num points.
        ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

        dY = np.zeros((2*tp, 1))                # Residuals.
        J  = np.zeros((2*tp, 6))                # Jacobian.

        #--- FILL ME IN ---
        Twc = Twcg
       
        for i in range(maxIters):
                R = Twc[:3, :3]
                t = Twc[:3, 3]
                for j in range(tp):
                        # e(x) = K * R^T * (Wpt - t) - Ipt
                        error = K @ R.T @ (Wpts[:,j]-t) 
                        error /= error[-1] 
  
                        e = error[:-1].reshape(2,1)
                        I = Ipts[:,j].reshape(2,1)
                        
                        dY[j:j+2,:] = e - I
                        
                        # compute jacobian matrix
                        Jacobian = find_jacobian(K, Twc, Wpts[:,j].reshape(3,1))
                        J[j:j+2,:] = Jacobian
                # solve normal equations
                dx = -inv(J.T @ J) @ J.T @ dY
                # converged? 
                if norm (dx) < 1e-4:
                        break
                # update parameters 
                x = epose_from_hpose(Twc)
                x+=dx
                Twc = hpose_from_epose(x)

        #------------------
        
        return Twc

#----- Functions Go Below -----

def epose_from_hpose(T):
        """Euler pose vector from homogeneous pose matrix."""
        E = np.zeros((6, 1))
        E[0:3] = np.reshape(T[0:3, 3], (3, 1))
        E[3:6] = rpy_from_dcm(T[0:3, 0:3])
        
        return E

def hpose_from_epose(E):
        """Homogeneous pose matrix from Euler pose vector."""
        T = np.zeros((4, 4))
        T[0:3, 0:3] = dcm_from_rpy(E[3:6])
        T[0:3, 3] = np.reshape(E[0:3], (3,))
        T[3, 3] = 1
        
        return T
