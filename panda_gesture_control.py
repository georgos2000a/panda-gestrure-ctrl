#!/usr/bin/env python3.9
import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
import pyrealsense2 as rs
import math
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header

# Demo - https://youtu.be/i1abYqUx0gw


# Initialize ROS node
rospy.init_node('panda_hand_gesture_control', anonymous=True)


# Initialize MoveIt! commander for the Panda arm
arm = MoveGroupCommander("panda_arm")
arm.set_max_velocity_scaling_factor(0.1)  # Set speed scaling
arm.set_max_acceleration_scaling_factor(0.1)  # Set acceleration scaling

# Initialize MoveIt! commander for the Panda hand/gripper
gripper = MoveGroupCommander("panda_hand")


class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        # Define resolution & frame rate
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30) #  Depth frame
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30) # Colour frame

        self.pipeline.start(config)

        # Select 'hand' preset from realsense presets
        depth_sensor = device.first_depth_sensor()
        if depth_sensor and depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.hand)

        # Create an align object
        # rs.stream.color indicates that we want to align depth to color
        self.align = rs.align(rs.stream.color)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)


        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return False, None, None

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())



        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()

def control_gripper(opening_width):
    # Initialize the MoveGroupCommander for the hand group
    gripper = MoveGroupCommander("panda_hand")

    # Calculate the joint values to achieve the desired opening width
    # The gripper consists of two fingers, each moving inwards or outwards equally.
    # The maximum width is 0.08 meters, so each joint moves 0.04 meters max.
    joint_value = opening_width / 2.0  # Each joint moves half of the total opening width

    # Set the joint target for the fingers
    gripper.set_joint_value_target([joint_value, joint_value])

    # Move the gripper to the desired joint value
    gripper.go(wait=True)

    # Stop and clear targets
    gripper.stop()
    gripper.clear_pose_targets()

# Initialize direction_history for smoother direction vector results
direction_history = deque(maxlen=5)  # Adjust maxlen as needed for smoothing

def calculate_direction_quaternion(wrist, midpoint):

    # Transform camera coordinates to chest coordinates
    wrist_world = np.array([   1300- wrist[2] , -(wrist[0] + 18) , -(wrist[1] -0.3)             ])
    midpoint_world = np.array([   1300- midpoint[2] , -(midpoint[0] + 18) , -(midpoint[1] -0.3)             ])


    # Ensure both wrist and midpoint are 3D vectors and convert to float64
    wrist = np.array(wrist, dtype=np.float64)
    midpoint = np.array(midpoint, dtype=np.float64)

    if wrist.shape[0] != 3 or midpoint.shape[0] != 3:
        raise ValueError("Both wrist and midpoint must be 3D coordinates.")

    # Calculate direction vector
    vector = wrist_world - midpoint_world

    # Normalize the vector vector
    vector /= np.linalg.norm(vector)

    direction = vector
    direction_history.append(direction)

    # Calculate the mean direction from history
    if len(direction_history) > 0:
        mean_direction = np.mean(direction_history, axis=0)
    else:
        mean_direction = np.array(direction)


    # Create a rotation matrix that aligns the Z-axis with the direction vector
    z_axis = mean_direction
    z_axis = [round(num, 1) for num in z_axis]

    x_axis = np.array([1, 0, 0], dtype=np.float64)

    # Avoid degenerate case where z_axis is too close to x_axis
    if np.allclose(z_axis, x_axis):
        x_axis = np.array([0, 1, 0], dtype=np.float64)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)

    rotation_matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], 0],
        [x_axis[1], y_axis[1], z_axis[1], 0],
        [x_axis[2], y_axis[2], z_axis[2], 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # Convert the rotation matrix to a quaternion
    quaternion = quaternion_from_matrix(rotation_matrix)
    # Normalize quaternion (length = 1)
    quaternion = quaternion / np.linalg.norm(quaternion)



    return quaternion



# Camera Intrinsics
focal_length = [590.6891  , 588.9445]
principal_point = [418.0332 , 254.8811]
skew = 0.1962

camera_matrix = np.array([[focal_length[0], skew, principal_point[0]],
                          [0, focal_length[1], principal_point[1]],
                          [0, 0, 1]])

# Distortion Coefficients
radial_distortion =[0.0672  , -0.1554]
tangential_distortion = [-0.0015 ,-0.0026]


dist_coeffs = np.array([radial_distortion[0], radial_distortion[1],
                        tangential_distortion[0], tangential_distortion[1], 0])



def rs2_deproject_pixel_to_point(intrin, pixel, depth):
    
    #Deprojects 2D pixel coordinates into 3D space using camera intrinsics and depth.
   
    x = (pixel[0] - intrin['ppx']) / intrin['fx']
    y = (pixel[1] - intrin['ppy']) / intrin['fy']

    if intrin['model'] == 'RS2_DISTORTION_INVERSE_BROWN_CONRADY':
        r2 = x * x + y * y
        f = 1 + intrin['coeffs'][0] * r2 + intrin['coeffs'][1] * r2 * r2 + intrin['coeffs'][4] * r2 * r2 * r2
        ux = x * f + 2 * intrin['coeffs'][2] * x * y + intrin['coeffs'][3] * (r2 + 2 * x * x)
        uy = y * f + 2 * intrin['coeffs'][3] * x * y + intrin['coeffs'][2] * (r2 + 2 * y * y)
        x = ux
        y = uy

    point = [depth * x, depth * y, depth]
    return point

# Prepare the intrinsics for the rs2_deproject_pixel_to_point function
intrin = {
    'fx': focal_length[0],         # Focal length in x direction
    'fy': focal_length[1],         # Focal length in y direction
    'ppx': principal_point[0],     # Principal point x
    'ppy': principal_point[1],     # Principal point y
    'model': 'RS2_DISTORTION_INVERSE_BROWN_CONRADY',  # Assuming Brown-Conrady model
    'coeffs': [radial_distortion[0], radial_distortion[1], tangential_distortion[0], tangential_distortion[1], 0]
}



# Initialize Mediapipe Hands and Holistic models
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Initialize Camera Intel RealSense
dc = DepthCamera()

# Initialize deques for smoothing wrist coordinates
right_wrist_px_history = deque(maxlen=5)
left_wrist_px_history = deque(maxlen=5)
robot_x_history= deque(maxlen=5)
robot_y_history= deque(maxlen=5)
robot_z_history= deque(maxlen=5)

while True:

    ret, depth_frame, frame = dc.get_frame()
    if not ret:
        break

    # Median filtering to remove zero values
    depth_frame[depth_frame == 0] = np.median(depth_frame[depth_frame != 0])

    # Undistort the RGB frame
    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Flip colour & depth frame
    undistorted_frame = cv2.flip(undistorted_frame, 1)  # Flip the frame horizontally
    depth_frame = cv2.flip(depth_frame, 1)  # Flip the depth frame horizontally

    img = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB for Mediapipe processing
    results = hands.process(img)  # Process the frame with Mediapipe Hands
    results2 = holistic.process(img)  # Process the frame using the Holistic model

    h, w, c = undistorted_frame.shape  # Get the height, width, and number of channels of the frame


    # Draw pose, face, and hands landmarks on the frame
    if results2.pose_landmarks:
        image_height, image_width, _ = img.shape

        # Right Wrist
        right_wrist = results2.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
        right_wrist_px = (int(right_wrist.x * image_width), int(right_wrist.y * image_height))
        right_wrist_px_history.append(right_wrist_px)

        # Left Wrist
        left_wrist = results2.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
        left_wrist_px = (int(left_wrist.x * image_width), int(left_wrist.y * image_height))
        left_wrist_px_history.append(left_wrist_px)

        # Compute smoothed coordinates
        smoothed_right_wrist_px = (
            int(np.mean([pt[0] for pt in right_wrist_px_history])),
            int(np.mean([pt[1] for pt in right_wrist_px_history]))
        )

        smoothed_left_wrist_px = (
            int(np.mean([pt[0] for pt in left_wrist_px_history])),
            int(np.mean([pt[1] for pt in left_wrist_px_history]))
        )

        # Check bounds to avoid IndexError and ensure hands are within frame
        if (0 <= smoothed_right_wrist_px[1] < depth_frame.shape[0] and
            0 <= smoothed_right_wrist_px[0] < depth_frame.shape[1]):
            right_wrist_depth = depth_frame[smoothed_right_wrist_px[1], smoothed_right_wrist_px[0]]
        else:
            right_wrist_depth = 0


        if (0 <= smoothed_left_wrist_px[1] < depth_frame.shape[0] and
            0 <= smoothed_left_wrist_px[0] < depth_frame.shape[1]):
            left_wrist_depth = depth_frame[smoothed_left_wrist_px[1], smoothed_left_wrist_px[0]]
        else:
            left_wrist_depth = 0



        mp.solutions.drawing_utils.draw_landmarks(undistorted_frame, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    if results.multi_hand_landmarks:
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            myHand = {}
            mylmList = []
            xList = []
            yList = []

            thumb_tip = None
            index_tip = None
            wrist = None
            index_pip = None

            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([id, px, py])
                xList.append(px)
                yList.append(py)

                # Extract thumb tip and index finger tip coordinates
                if id == mpHands.HandLandmark.THUMB_TIP:
                    thumb_tip = (px, py)
                elif id == mpHands.HandLandmark.INDEX_FINGER_TIP:
                    index_tip = (px, py)
                elif id == mpHands.HandLandmark.WRIST:  # Extract wrist coordinates
                    wrist = (px, py)
                elif id == mpHands.HandLandmark.INDEX_FINGER_PIP:  # Extract wrist coordinates
                   index_pip = (px, py)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)
            myHand["type"] = handType.classification[0].label

            myHand["thumb_tip"] = thumb_tip
            myHand["index_tip"] = index_tip
            myHand["wrist"] = wrist
            myHand["index_pip"] = index_pip


            if thumb_tip is not None and index_tip is not None:
                thumb_tip_depth = depth_frame[thumb_tip[1], thumb_tip[0]]
                index_tip_depth = depth_frame[index_tip[1], index_tip[0]]
                wrist_depth = depth_frame[wrist[1],wrist[0]]
                index_pip_depth = depth_frame[index_pip[1],index_pip[0]]

                # Get point coordinates for the camera coordinate system
                thumb_3d = rs2_deproject_pixel_to_point(intrin,pixel = thumb_tip, depth = thumb_tip_depth)
                index_3d = rs2_deproject_pixel_to_point(intrin,pixel =index_tip, depth = index_tip_depth)
                wrist_3d = rs2_deproject_pixel_to_point(intrin,pixel = wrist, depth = wrist_depth)
                index_pip_3d = rs2_deproject_pixel_to_point(intrin,pixel = index_pip, depth = index_pip_depth)

                #------- THIS SECTION IS FOR GRIPPER OPENING CONTROLLING -------------
                # Calculate the distance between thum and index tip in mm
                distance = math.sqrt(
                    (index_3d[0] - thumb_3d[0]) ** 2 +
                    (index_3d[1] - thumb_3d[1]) ** 2 +
                    (index_3d[2] - thumb_3d[2]) ** 2
                )

                # Convert distance in meters
                distance = distance / 1000

                # Round to 3 decimals for millimeter-level precision
                distance = round(distance, 3)

                # Connect hand distance values to match with gripper opening limits
                if distance <= 0.08 :
                    gripper_width = distance
                # When hand was fully closed or fully open distance values were getting
                # values around 60m. A open hand could not be larger than 0.5m
                # So this we added an extra if condition
                elif distance > 0.08 and distance <= 0.5 :
                    gripper_width = 0.08
                else :
                    gripper_width = 0

                # Call control gripper function
                control_gripper(gripper_width)

                #-------END OF GRIPPER OPENING CONTROLLING SECTION---------------------


                # Calculate midpoint between thumb tip & index tip
                # Integer x , y because the values are refering to pixels
                midpoint_x = (thumb_tip[0] + index_tip[0]) // 2 # // is integer division
                midpoint_y = (thumb_tip[1] + index_tip[1]) // 2
                midpoint = (midpoint_x, midpoint_y)


                midpoint_depth = (thumb_tip_depth + index_tip_depth) / 2
                print(f"Midpoint between Thumb Tip and Index Finger Tip: {midpoint}")
                print(midpoint_depth)


                # Locate midpoint in real world in respect to camera's axes
                point_3d = rs2_deproject_pixel_to_point(intrin,pixel = midpoint, depth = midpoint_depth)

                # Convert cameras coordinates to chest axes
                world_x = 1300- point_3d[2]
                world_y = -(point_3d[0] + 18)
                world_z = -(point_3d[1] -0.3)

                """
                Robot's end effector maximum distance in every axis (base coordinate system) , in meters :
                x = +/- 0.77m , positive to the front
                y =  (-0.7 , 0.7), positive to the left
                z = (-0.1 , +0.9), positive upwards

                User's hand  maximum distance in every axis (chest coordinate system) , in mm :
                x = +/- 600mm , positive to the front
                y =  (-700 , 700), positive to the left
                z = (-700 , +700), positive upwards
                """

                # Calculate axis-x coordinates for the robot
                if world_x >=-600 and world_x <= 600 :
                    robot_x = world_x * (0.77 / 600)
                elif world_x > 600 :
                    robot_x =  0.77
                else :
                    robot_x = -0.77


                # Calculate axis-y coordinates for the robot
                if world_y >=-700 and world_y <= 700 :
                    robot_y = world_y * (0.7 / 700)
                elif world_y > 700 :
                    robot_y =  0.7
                else :
                    robot_y = -0.7

                # Calculate axis-z coordinates for the robot
                if world_z >= 0 and world_z <= 700 :
                    robot_z = world_z * (0.9 / 700)
                elif world_z > 700:
                    robot_z = 0.9
                elif world_z < 0 and world_z > -350 :
                    robot_z = world_z * (-0.1 / -350)
                else :
                    robot_z = -0.1


                robot_x_history.append(robot_x)
                robot_y_history.append(robot_y)
                robot_z_history.append(robot_z)

                # Calculate the smoothed coordinates by averaging the history
                robot_x = np.mean(robot_x_history)
                robot_y = np.mean(robot_y_history)
                robot_z = np.mean(robot_z_history)

                # Round smoothed coordinates to 2 decimal points
                robot_x = round(robot_x, 2)
                robot_y = round(robot_y, 2)
                robot_z = round(robot_z, 2)



                # Calculate quaternion from wrist to index_pip
                quaternion = calculate_direction_quaternion(wrist_3d, index_pip_3d)



                # Move the robot arm to the target position
                pose_target = arm.get_current_pose().pose

                pose_target.orientation.x = quaternion[0]
                pose_target.orientation.y = quaternion[1]
                pose_target.orientation.z = quaternion[2]
                pose_target.orientation.w = quaternion[3]


                pose_target.position.x =  robot_x
                pose_target.position.y = robot_y
                pose_target.position.z = robot_z

                print(pose_target.position.x , pose_target.position.y , pose_target.position.z)

                arm.set_pose_target(pose_target)
                arm.go(wait=True)
                arm.stop()
                arm.clear_pose_targets()


            # Optionally, print the entire coordinate for verification
            print(f"Thumb Tip: {myHand['thumb_tip']}")
            print(f"Index Finger Tip: {myHand['index_tip']}")

            mpdraw.draw_landmarks(undistorted_frame, handLms, mpHands.HAND_CONNECTIONS)
            cv2.rectangle(undistorted_frame, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)
            cv2.putText(undistorted_frame, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)




    cv2.imshow("Depth Frame", depth_frame)
    cv2.imshow("Color Frame", undistorted_frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to close application
        break

# Release resources
dc.release()
cv2.destroyAllWindows()

