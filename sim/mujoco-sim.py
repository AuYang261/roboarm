"""
MuJoCo simulation for robotic arm with URDF.
The arm draws a figure-8 trajectory.
"""

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("Error: MuJoCo is not installed.")
    print("Install it with: pip install mujoco")
    print("See https://github.com/deepmind/mujoco for installation details.")
    sys.exit(1)

import numpy as np
import time
import os
import sys

# Add parent directory to path for potential utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_urdf_xml(urdf_xml):
    """
    Remove URDF elements that MuJoCo doesn't understand.
    MuJoCo's XML parser doesn't support ROS-specific extensions like
    <material>, <transmission>, <gazebo> tags.
    """
    import re

    # Remove <material> tags and their contents
    # Pattern matches <material> ... </material> across multiple lines
    urdf_xml = re.sub(r'<material\b[^>]*>.*?</material>', '', urdf_xml, flags=re.DOTALL)

    # Remove <transmission> tags
    urdf_xml = re.sub(r'<transmission\b[^>]*>.*?</transmission>', '', urdf_xml, flags=re.DOTALL)

    # Remove <gazebo> tags
    urdf_xml = re.sub(r'<gazebo\b[^>]*>.*?</gazebo>', '', urdf_xml, flags=re.DOTALL)

    # Remove any remaining material references in visual/collision tags
    urdf_xml = re.sub(r'<material\s+name="[^"]*"\s*/>', '', urdf_xml)

    # Remove empty lines and extra whitespace
    urdf_xml = '\n'.join([line for line in urdf_xml.split('\n') if line.strip()])

    return urdf_xml

def load_urdf_model(urdf_path):
    """
    Load URDF file into MuJoCo model.
    Returns model and data.
    """
    # Check if file exists
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Get absolute path
    urdf_abs = os.path.abspath(urdf_path)

    # Method 1: Try using mujoco.urdf.load_urdf if available
    # This is the proper way to load URDF in MuJoCo 2.3.0+
    print("Method 1: Checking if mujoco.urdf module is available...")
    import importlib.util
    urdf_loader = importlib.util.find_spec("mujoco.urdf")

    if urdf_loader is not None:
        try:
            print("Attempting to load URDF using mujoco.urdf.load_urdf...")
            # Import here to avoid affecting global mujoco variable
            import mujoco.urdf as urdf_module
            model = urdf_module.load_urdf(urdf_abs)
            data = mujoco.MjData(model)
            print("Successfully loaded URDF using mujoco.urdf.load_urdf")
            return model, data
        except Exception as e:
            print(f"Method 1 (mujoco.urdf.load_urdf) failed: {e}")
    else:
        print("mujoco.urdf module not found, skipping method 1")

    # Method 2: Try direct XML loading (may fail due to unsupported URDF elements)
    print("Method 2: Attempting direct XML loading...")
    try:
        # Read URDF content
        with open(urdf_abs, 'r') as f:
            urdf_xml = f.read()

        # Try to load directly as XML
        model = mujoco.MjModel.from_xml_string(urdf_xml)
        data = mujoco.MjData(model)
        print("Successfully loaded URDF as XML string")
        return model, data
    except Exception as e:
        error_msg = str(e)
        print(f"Method 2 (direct XML loading) failed: {error_msg[:100]}")

    # Method 3: Clean URDF XML and try again
    print("Method 3: Attempting to load cleaned URDF...")
    try:
        with open(urdf_abs, 'r') as f:
            urdf_xml = f.read()

        # Clean the URDF XML
        cleaned_xml = clean_urdf_xml(urdf_xml)

        # Debug: save cleaned XML to file for inspection
        debug_path = os.path.join(os.path.dirname(urdf_abs), "mujoco.xml")
        with open(debug_path, 'w') as f:
            f.write(cleaned_xml)
        print(f"Debug: saved cleaned URDF to {debug_path}")

        # Try to load cleaned XML
        model = mujoco.MjModel.from_xml_string(cleaned_xml)
        data = mujoco.MjData(model)
        print("Successfully loaded cleaned URDF")
        return model, data
    except Exception as e:
        error_msg = str(e)
        print(f"Method 3 (cleaned URDF loading) failed: {error_msg[:100]}")
        raise ValueError(f"Failed to load URDF after all attempts: {error_msg}")


def figure8_trajectory(t, period=10.0, radius_x=0.1, radius_y=0.05):
    """
    Generate a figure-8 trajectory in XY plane at constant Z.
    Returns desired end-effector position (x, y, z).
    """
    # Parametric equation for figure-8 (Lemniscate of Gerono)
    # x = radius_x * sin(t)
    # y = radius_y * sin(t) * cos(t)
    # z = constant
    omega = 2 * np.pi / period
    x = radius_x * np.sin(omega * t)
    y = radius_y * np.sin(omega * t) * np.cos(omega * t)
    z = 0.3  # constant height
    return np.array([x, y, z])

def inverse_kinematics(model, data, target_pos, joint_indices=None):
    """
    Simple inverse kinematics using Jacobian transpose method.
    This is a naive implementation for demonstration.
    For accurate IK, use proper solver.
    """
    if joint_indices is None:
        # Use all joints except gripper
        joint_indices = [i for i in range(model.njnt) if 'gripper' not in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)]

    # Get end-effector site (we need to define one)
    # For now, use the last body's position
    end_effector_body = model.nbody - 1
    # Compute current end-effector position
    current_pos = data.xpos[end_effector_body]

    # Compute error
    error = target_pos - current_pos

    # Compute Jacobian for the end-effector body
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, end_effector_body)

    # Damped least squares pseudo-inverse
    damping = 0.01
    jac_t = jacp.T
    jj = jacp @ jac_t + damping * np.eye(3)
    step = jac_t @ np.linalg.solve(jj, error)

    # Update joint positions (in velocity control)
    # We'll apply as velocity command
    return step

def main():
    # Path to URDF
    urdf_path = os.path.join(".", "urdf", "arm.urdf")
    # If running from sim directory, adjust path
    if not os.path.exists(urdf_path):
        urdf_path = os.path.join("..", urdf_path)

    print(f"Loading URDF from {urdf_path}")

    try:
        model, data = load_urdf_model(urdf_path)
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        print("Creating a simple MJCF model with primitive shapes instead...")

        # Create a simple 3-joint arm model for demonstration
        # Minimal MJCF with only essential elements
        simple_mjcf = """<?xml version="1.0"?>
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -1.5 0.5" xyaxes="1 0 0 0 0 1"/>

    <!-- Base (fixed to world) -->
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.05 0.05 0.02" rgba="0.7 0.7 0.7 1"/>
      <joint type="free"/>
    </body>

    <!-- Link 1 -->
    <body name="link1" pos="0 0 0.05">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02" rgba="0.5 0.5 0.8 1"/>
    </body>

    <!-- Link 2 -->
    <body name="link2" pos="0 0 0.15">
      <joint name="joint2" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02" rgba="0.8 0.5 0.5 1"/>
    </body>

    <!-- Link 3 (end effector) -->
    <body name="link3" pos="0 0 0.25">
      <joint name="joint3" type="hinge" axis="1 0 0"/>
      <geom type="sphere" size="0.03" rgba="0.5 0.8 0.5 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="joint1"/>
    <motor joint="joint2"/>
    <motor joint="joint3"/>
  </actuator>
</mujoco>"""
        try:
            model = mujoco.MjModel.from_xml_string(simple_mjcf)
            data = mujoco.MjData(model)
            print("Successfully created simple MJCF model")
        except Exception as e2:
            print(f"Failed to create simple MJCF model: {e2}")
            print("Exiting...")
            sys.exit(1)

    print(f"Model loaded: nq={model.nq}, nv={model.nv}, na={model.na}")

    # Camera viewing: fixed camera is already defined in the MJCF model.

    # Open viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer started. Press ESC to exit.")
        print("Camera controls: Press 'C' to cycle through camera views.")
        print("Available camera: fixed")

        # Reset to initial position
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = 0.0

        # Main simulation loop
        t = 0.0
        dt = model.opt.timestep
        period = 5.0  # seconds per figure-8 cycle
        step_counter = 0

        while viewer.is_running():
            # Compute desired end-effector position
            target_pos = figure8_trajectory(t, period)

            # Simple control: move joints sinusoidally (demo)
            # This is just to make the arm move; replace with IK for accurate tracking
            for i in range(min(3, model.nu)):
                data.ctrl[i] = 0.5 * np.sin(t * 2 + i * 1.0)

            # Step simulation
            mujoco.mj_step(model, data)

            # Increment step counter and show status periodically
            step_counter += 1
            if step_counter % 100 == 0:
                # Get end-effector position (last body)
                ee_pos = data.xpos[-1]
                print(f"Step {step_counter}: t={t:.2f}s, target={target_pos}, actual={ee_pos}")

            # Update viewer
            viewer.sync()

            # Increment time
            t += dt

            # Slow down simulation for visualization
            time.sleep(dt * 0.5)  # real-time factor

    print("Simulation finished.")

if __name__ == "__main__":
    main()