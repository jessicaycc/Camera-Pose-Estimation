import numpy as np

def find_jacobian(K, Twc, Wpt):
        """
        Determine the Jacobian for NLS camera pose optimization.

        The function computes the Jacobian of an image plane point with respect
        to the current camera pose estimate, given a landmark point. The 
        projection model is the simple pinhole model.

        Parameters:
        -----------
        K    - 3x3 np.array, camera intrinsic calibration matrix.
        Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
        Wpt  - 3x1 world point on calibration target (one of n).

        Returns:
        --------
        J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
        """
        #--- FILL ME IN ---
        
        # Extract rotation and translation matrix  
        J = []
        R = Twc[:3, :3]
        t = Twc[:3,3:4]

        # Convert to euler angles 
        rpy = rpy_from_dcm(R)
        r = float(rpy[0])
        p = float(rpy[1])
        y = float(rpy[2])

        # Cross Matrix 
        I1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        I2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        I3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
       
        # partial derivative of rotational matrix wrt roll pitch yaw 
        Rr = dcm_from_rpy((r, 0, 0))
        Rp = dcm_from_rpy((0, p, 0))
        Ry = dcm_from_rpy((0, 0, y))
        
        dRdr = Ry @ Rp @ I1 @ Rr
        dRdp = Ry @ I2 @ Rp @ Rr
        dRdy = I3 @ Ry @ Rp @ Rr

        # partial derivative of translation wrt xyz
        dxdt = np.array([[ -1,  0,  0], 
                         [  0, -1,  0],
                         [  0,  0, -1]]) 

        # dx/d<rpy>
        roll = K @ dRdr.T @ (Wpt - t)
        pitch = K @ dRdp.T @ (Wpt - t)
        yaw = K @ dRdy.T @ (Wpt - t)
        dxdeuler = np.concatenate((roll, pitch, yaw), axis = 1)
       
        # dx/d<xyz>
        dxdcoord = K @ R.T @ dxdt
        J = np.concatenate((dxdcoord,dxdeuler), axis=1)
       
        x = K @ R.T @ (Wpt - t)
        
        x_ = x[-1]
        J_ = J[-1:,:]
        
        J = (J * x_ - x @ J_) / np.power(x_,2)
        J = J[0:2]


        #------------------

        return J

def rpy_from_dcm(R):
        """
        Roll, pitch, yaw Euler angles from rotation matrix.

        The function computes roll, pitch and yaw angles from the
        rotation matrix R. The pitch angle p is constrained to the range
        (-pi/2, pi/2].  The returned angles are in radians.

        Inputs:
        -------
        R  - 3x3 orthonormal rotation matrix.

        Returns:
        --------
        rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
        """
        rpy = np.zeros((3, 1))

        # Roll.
        rpy[0] = np.arctan2(R[2, 1], R[2, 2])

        # Pitch.
        sp = -R[2, 0]
        cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

        if np.abs(cp) > 1e-15:
                rpy[1] = np.arctan2(sp, cp)
        else:
        # Gimbal lock...
                rpy[1] = np.pi/2
        
                if sp < 0:
                        rpy[1] = -rpy[1]

        # Yaw.
        rpy[2] = np.arctan2(R[1, 0], R[0, 0])

        return rpy

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])