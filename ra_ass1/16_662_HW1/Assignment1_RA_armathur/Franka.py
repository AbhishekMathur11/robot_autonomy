import numpy as np
import RobotUtil as rt
import math

class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity - NOTE: don't change
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices - NOTE: don't change this part
        self.Tlink = []  # Transforms for each link (const)
        self.Tjoint = []  # Transforms for each joint (init eye)
        self.Tcurr = []  # Coordinate frame of current (init eye)
        
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.], [0, 0, 0, 1]])
            self.Tjoint.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.]
        self.ForwardKin([0., 0., 0., 0., 0., 0., 0.])

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''

        self.q = ang

        for i in range(7):
            self.Tjoint[i] = rt.rpyxyz2H(ang[i]*np.array(self.axis[i]),[0,0,0])
        
        for i in range(7):
            if i == 0:
                self.Tcurr[i] = self.Tlink[i] @ self.Tjoint[i]
            else:
                self.Tcurr[i] = self.Tcurr[i-1] @ self.Tlink[i] @ self.Tjoint[i]
            
        # Compute current joint and end effector coordinate frames (self.Tjoint). Remember that not all joints rotate about the z axis!
        
        self.Tcurr[7] = self.Tcurr[6] @ self.Tlink[7]


        for i in range(7):
            p = self.Tcurr[7][:3, 3] - self.Tcurr[i][:3, 3]

            z_i = self.Tcurr[i][:3, 2]  
            self.J[0:3, i] = np.cross(z_i, p)
            self.J[3:6, i] = z_i
        return self.Tcurr, self.J

    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose, 
        Error in your IK solution compared to the desired target
        '''
        C = np.array([[1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1000.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1000.0]
                        ])

        W = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
                    ])
        max_iter = 10000
        


        q = np.array(ang)

        for i in range(max_iter):

            T_curr, J = self.ForwardKin(q)

            Rcurr = T_curr[-1][:3, :3]
            pcurr = T_curr[-1][:3, 3]

            Rgoal = TGoal[:3, :3]
            pgoal = TGoal[:3, 3]

            e_p = pgoal - pcurr
            
            R_diff = Rgoal @ Rcurr.T
            axis, angle = rt.R2axisang(R_diff)
            e_R = angle * np.array(axis)

            e = np.concatenate([e_p, e_R])

            if np.linalg.norm(e_p) < x_eps and np.linalg.norm(e_R) < r_eps:
                break

            W_inv = np.linalg.inv(W)  
            C_inv = np.linalg.inv(C)  

          
            J_hash = W_inv @ J.T @ np.linalg.inv(J @ W_inv @ J.T + C_inv)

            dq = J_hash @ e
            q += dq 

        e_p_final = pgoal - T_curr[-1][:3, 3]
        R_diff_final = np.dot(Rgoal, T_curr[-1][:3, :3].T)
        axis_final, angle_final = rt.R2axisang(R_diff_final)
        e_R_final = angle_final * np.array(axis_final)

        
        Err = np.concatenate([e_p_final, e_R_final])

        return q, Err
