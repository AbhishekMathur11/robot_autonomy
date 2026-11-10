import mujoco as mj
from mujoco import viewer
import numpy as np
import math
#import quaternion


# Set the XML filepath
xml_filepath = "/home/abhishek/ra_ass1/16_662_HW1/franka_emika_panda/panda_nohand_torque_fixed_board.xml"

################################# Control Callback Definitions #############################

# Control callback for gravity compensation
def gravity_comp(model, data):
    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data):  # TODO:
    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    f_des = np.array([15.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Get the Jacobian for the desired location on the robot (The end-effector)
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))

    mj.mj_jacBody(model, data, jac_p, jac_r, model.body("hand").id)

    
    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables

    # Specify the desired force in global coordinates
    jac = np.vstack((jac_p, jac_r))
    # Compute the required control input using desied force values

    tau_des = jac.T @ f_des

    # Set the control inputs

    data.ctrl[:7] = tau_des[:7] + data.qfrc_bias[:7]

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    global force
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

# Control callback for an impedance controller
def impedance_control(model, data):  # TODO:

    # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mj.mj_jacBody(model, data, jac_p, jac_r, model.body("hand").id)

    jac = np.vstack((jac_p, jac_r))

    # Instantite a handle to the desired body on the robot
    ee_pos = data.xpos[model.body("hand").id]
    ee_vel = jac @ data.qvel

    # Set the desired position
    pos_des = np.array([15, 0, 0.5])
    vel_des = np.zeros(6)

    pos_error = pos_des - ee_pos
    vel_error = vel_des - ee_vel

    Kp = np.diag([1, 1, 1, 1, 1, 1])
    Kd = np.diag([1, 1, 1, 1, 1, 1])

    # f_des = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    f_imp = Kp @ np.concatenate([pos_error, np.zeros(3)]) + Kd @ vel_error
    f_total = f_imp 
    tau_des = jac.T @ f_total

    data.ctrl[:7] = tau_des[:7] + data.qfrc_bias[:7]

    
    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    global force
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]


def position_control(model, data):
    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    # desired_joint_positions = np.array(
        # [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    desired_joint_positions = np.array([0.17964678,  0.25731503, -0.20852791, -2.08281856, -0.03273823,  3.90704583, -3.06493904])

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = 430

    # Desired gain on velocity error (K_d)
    Kd = 430

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp * \
        (desired_joint_positions-data.qpos[:7]) + Kd * \
        (np.array([0, 0, 0, 0, 0, 0, 0])-data.qvel[:7])


####################################### MAIN #####################################
if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    mj.set_mjcb_control(impedance_control)  # TODO:

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(
        force_sensor_max_time/model.opt.timestep))

    # Launch the simulate viewer
    viewer.launch(model, data)

    # Save recorded force and time points as a csv file
    force = np.reshape(force, (5000, 1))
    time = np.reshape(time, (5000, 1))
    plot = np.concatenate((time, force), axis=1)
    np.savetxt('impedance_control_1.csv', plot, delimiter=',')
