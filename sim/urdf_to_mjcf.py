#!/usr/bin/env python3
"""
URDF to MuJoCo MJCF converter.

Converts URDF files to MuJoCo's MJCF format by:
1. Removing ROS-specific tags (material, transmission, gazebo)
2. Converting URDF structure to MJCF structure
3. Adding necessary MJCF elements (compiler, worldbody, actuators)
"""

import os
import sys
import re
import argparse
from pathlib import Path

def clean_urdf_xml(urdf_xml, urdf_dir=None):
    """
    Remove URDF elements that MuJoCo doesn't understand.
    MuJoCo's XML parser doesn't support ROS-specific extensions like
    <material>, <transmission>, <gazebo> tags.
    Also convert relative mesh paths to absolute paths if urdf_dir is provided.
    """
    # Remove <material> tags and their contents
    urdf_xml = re.sub(r'<material\b[^>]*>.*?</material>', '', urdf_xml, flags=re.DOTALL)

    # Remove <transmission> tags
    urdf_xml = re.sub(r'<transmission\b[^>]*>.*?</transmission>', '', urdf_xml, flags=re.DOTALL)

    # Remove <gazebo> tags
    urdf_xml = re.sub(r'<gazebo\b[^>]*>.*?</gazebo>', '', urdf_xml, flags=re.DOTALL)

    # Remove any remaining material references in visual/collision tags
    urdf_xml = re.sub(r'<material\s+name="[^"]*"\s*/>', '', urdf_xml)

    # Convert mesh paths to absolute paths if urdf_dir is provided
    if urdf_dir is not None:
        urdf_abs_dir = os.path.abspath(urdf_dir)

        def replace_mesh_path(match):
            filename = match.group(1)
            if not os.path.isabs(filename):
                abs_path = os.path.join(urdf_abs_dir, filename)
            else:
                abs_path = filename
                try:
                    rel_path = os.path.relpath(abs_path, urdf_abs_dir)
                    if not rel_path.startswith('..'):
                        abs_path = rel_path
                except ValueError:
                    pass

            abs_path = os.path.normpath(abs_path)
            if not os.path.exists(os.path.join(urdf_abs_dir, abs_path) if not os.path.isabs(abs_path) else abs_path):
                print(f"Warning: Mesh file not found: {abs_path}")
            abs_path = abs_path.replace('\\', '/')
            return f'filename="{abs_path}"'

        urdf_xml = re.sub(r'filename="([^"]+\.(?:stl|dae|obj|STL|DAE|OBJ))"',
                         replace_mesh_path, urdf_xml, flags=re.IGNORECASE)

    # Remove empty lines and extra whitespace
    urdf_xml = '\n'.join([line for line in urdf_xml.split('\n') if line.strip()])

    return urdf_xml

def extract_joints_from_urdf(urdf_xml):
    """
    Extract joint names from URDF XML.
    Returns list of joint names.
    """
    joint_pattern = r'<joint\s+name="([^"]+)"'
    joints = re.findall(joint_pattern, urdf_xml)
    return joints

def clean_urdf(urdf_path, output_path=None):
    """
    Clean URDF file by removing MuJoCo-unsupported tags.
    Returns path to cleaned URDF file.
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Set default output path if not provided
    if output_path is None:
        output_path = urdf_path.with_suffix('.cleaned.xml')
    output_path = Path(output_path)

    # Read URDF content
    with open(urdf_path, 'r') as f:
        urdf_xml = f.read()

    # Clean URDF XML
    print(f"Cleaning URDF XML...")
    cleaned_urdf = clean_urdf_xml(urdf_xml, urdf_path.parent)

    # Save cleaned URDF
    with open(output_path, 'w') as f:
        f.write(cleaned_urdf)

    print(f"Successfully cleaned URDF:")
    print(f"  Input:  {urdf_path}")
    print(f"  Output: {output_path}")

    return output_path

def convert_urdf_to_mjcf(urdf_path, output_path=None, add_actuators=True):
    """
    Convert URDF file to MuJoCo MJCF format.

    Args:
        urdf_path: Path to input URDF file
        output_path: Path to output MJCF file (default: same directory with .xml extension)
        add_actuators: Whether to add motor actuators for each joint
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Set default output path if not provided
    if output_path is None:
        output_path = urdf_path.with_suffix('.mujoco.xml')
    output_path = Path(output_path)

    # Read URDF content
    with open(urdf_path, 'r') as f:
        urdf_xml = f.read()

    # Clean URDF XML
    print(f"Cleaning URDF XML...")
    cleaned_urdf = clean_urdf_xml(urdf_xml, urdf_path.parent)

    # Extract joint names for actuators
    joints = extract_joints_from_urdf(cleaned_urdf)
    print(f"Found {len(joints)} joints: {joints}")

    # Extract content inside <robot> tags (remove XML declaration and robot tags)
    # Remove XML declaration if present
    robot_content = re.sub(r'^<\?xml[^?>]*\?>', '', cleaned_urdf, flags=re.MULTILINE)

    # Extract content between <robot> and </robot> tags
    robot_match = re.search(r'<robot[^>]*>(.*?)</robot>', robot_content, re.DOTALL)
    if robot_match:
        robot_inner_content = robot_match.group(1).strip()
    else:
        # If no robot tags, use the whole content
        robot_inner_content = robot_content.strip()
        print("Warning: No <robot> tags found in cleaned URDF")

    # Create MJCF XML structure
    mjcf_template = f"""<?xml version="1.0"?>
<mujoco model="{urdf_path.stem}">
  <!-- Compiler settings -->
  <compiler angle="radian" inertiafromgeom="true" settotalmass="14"/>
  <option timestep="0.01" integrator="RK4"/>

  <!-- Default settings -->
  <default>
    <geom contype="1" conaffinity="1" condim="3" friction="0.7 0.1 0.1"/>
    <joint armature="0.01" damping="0.1" limited="false"/>
    <motor ctrlrange="-1.0 1.0" ctrllimited="true"/>
  </default>

  <!-- Assets - mesh files will be referenced from URDF -->
  <asset>
    <!-- Mesh files are referenced directly in the URDF content -->
  </asset>

  <!-- World body containing the robot -->
  <worldbody>
    <!-- Floor for visualization -->
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>
    <light name="top" pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -1.5 0.5" xyaxes="1 0 0 0 0 1"/>

    <!-- Base link fixed to world -->
    <body name="world" pos="0 0 0">
      <freejoint name="world_free"/>
    </body>

    <!-- Insert cleaned URDF robot here -->
    {robot_inner_content}
  </worldbody>

  <!-- Actuators -->
  <actuator>
"""

    # Add motor actuators for each joint
    if add_actuators and joints:
        for joint in joints:
            mjcf_template += f'    <motor name="{joint}_motor" joint="{joint}"/>\n'

    mjcf_template += """  </actuator>

  <!-- Contact settings -->
  <contact>
    <exclude body1="world" body2="base_link"/>
  </contact>

</mujoco>"""

    # Save MJCF XML
    with open(output_path, 'w') as f:
        f.write(mjcf_template)

    print(f"Successfully converted URDF to MJCF:")
    print(f"  Input:  {urdf_path}")
    print(f"  Output: {output_path}")
    print(f"  Joints with actuators: {len(joints)}")

    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Convert URDF to MuJoCo-compatible XML formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s urdf/arm.urdf                   # Clean URDF (default mode)
  %(prog)s urdf/arm.urdf -o cleaned.xml    # Clean URDF with custom output
  %(prog)s urdf/arm.urdf --mode mjcf       # Convert to full MJCF format
  %(prog)s urdf/arm.urdf --mode mjcf --no-actuators  # MJCF without actuators
""")
    parser.add_argument('input', help='Input URDF file path')
    parser.add_argument('-o', '--output', help='Output XML file path')
    parser.add_argument('--mode', choices=['clean', 'mjcf'], default='clean',
                       help='Conversion mode: clean (remove unsupported tags) or mjcf (full MJCF format)')
    parser.add_argument('--no-actuators', action='store_true',
                       help='Do not add motor actuators (MJCF mode only)')

    args = parser.parse_args()

    try:
        if args.mode == 'clean':
            output_path = clean_urdf(args.input, args.output)
            print(f"\nCleaning successful!")
            print(f"To use in MuJoCo: model = mujoco.MjModel.from_xml_path('{output_path}')")
        else:  # mjcf mode
            output_path = convert_urdf_to_mjcf(
                args.input,
                args.output,
                add_actuators=not args.no_actuators
            )
            print(f"\nMJCF conversion successful!")
            print(f"To use in MuJoCo: model = mujoco.MjModel.from_xml_path('{output_path}')")
    except Exception as e:
        print(f"Error processing URDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()