#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class TurtleBotFollower:
    def __init__(self):
        rospy.init_node('turtlebot_follower', anonymous=True)

        # Publishers and subscribers
        self.pub_cmd_vel = rospy.Publisher('/follower/cmd_vel', Twist, queue_size=10)
        self.rgb_sub = rospy.Subscriber('/follower/image_raw', Image, self.rgb_camera_callback)
        self.depth_sub = rospy.Subscriber('/follower/depth/image_raw', Image, self.depth_camera_callback)
        self.bridge = CvBridge()

        # PD control gains for distance
        self.kp_dist = 0.5
        self.kd_dist = 0.05

        # PD control gains for angular control
        self.kp_angle = 0.3
        self.kd_angle = 0.005

        self.desired_distance = 0.3  # Desired distance from the leader (in meters)
        self.previous_distance_error = 0.0
        self.previous_angle_error = 0.0
        self.cmd_vel = Twist()
        self.rate = rospy.Rate(10)  # Loop rate: 10 Hz
        self.last_time = rospy.get_time()

        self.red_contours = []  # Store red contours
        self.depth_image = None
        self.frame_width = None
        self.angular_tolerance = 0.05  # Ignore small angular errors

        # Variables for smoothing
        self.cx_rolling_avg = None
        self.alpha = 0.5  # Low-pass filter factor for smoothing

    def rgb_camera_callback(self, image_msg):
        try:
            # Convert ROS image to OpenCV format
            rgb_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Store frame dimensions for later use
            if self.frame_width is None:
                self.frame_width = rgb_image.shape[1]

            # Detect red color in the image
            mask, self.red_contours = self.detect_red_regions(rgb_image)

            # Visualize the red detection for debugging
            if mask is not None:
                cv2.imshow("Red Mask", mask)
                cv2.waitKey(1)

            if self.red_contours:
                rospy.loginfo(f"Contours detected: {len(self.red_contours)}")
                for contour in self.red_contours:
                    cv2.drawContours(rgb_image, [contour], -1, (0, 255, 0), 2)
            else:
                rospy.logwarn("No red contours detected")

            cv2.imshow("RGB Image with Red Detection", rgb_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Error processing RGB image: {e}")

    def depth_camera_callback(self, image_msg):
        try:
            # Convert ROS image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="32FC1")
            self.depth_image[self.depth_image <= 0] = np.nan  # Filter invalid values
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
            return

        # Process the red regions and determine the depth
        if self.red_contours and self.depth_image is not None:
            selected_depth, angular_error = self.get_closest_red_depth_and_orientation()
            if selected_depth is not None:
                self.maintain_distance(selected_depth, angular_error)
            else:
                rospy.logwarn("No valid depth for detected red regions")
                self.stop_follower()
        else:
            self.stop_follower()

    def detect_red_regions(self, image):
        """Detects all red regions in the given RGB image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # Find contours in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Return the mask and contours for debugging
        return red_mask, contours

    def get_closest_red_depth_and_orientation(self):
        """Find the closest red region based on depth and calculate angular error and orientation."""
        min_depth = float('inf')
        closest_depth = None
        angular_error = None
        aspect_ratio = None

        for contour in self.red_contours:
            if cv2.contourArea(contour) < 500:  # Ignore small contours to reduce noise
                continue

            # Get the bounding box and calculate width and height
            rect = cv2.minAreaRect(contour)  # Get rotated rectangle
            (cx, cy), (width, height), angle = rect

            # Calculate aspect ratio
            aspect_ratio = max(width, height) / min(width, height)

            # Determine if the leader is facing or turned
            if aspect_ratio < 1.2:  # Close to square
                rospy.loginfo("Leader is facing directly.")
            else:
                rospy.loginfo("Leader is turned sideways.")

            # Smoothing cx using a low-pass filter
            if self.cx_rolling_avg is None:
                self.cx_rolling_avg = cx  # Initialize rolling average
            else:
                self.cx_rolling_avg = self.alpha * cx + (1 - self.alpha) * self.cx_rolling_avg

            # Calculate angular error as the horizontal offset from the image center
            frame_center = self.frame_width / 2
            angular_error = (self.cx_rolling_avg - frame_center) / frame_center  # Normalize to [-1, 1]

            # Validate if within depth image bounds
            cx, cy = int(cx), int(cy)
            if 0 <= cx < self.depth_image.shape[1] and 0 <= cy < self.depth_image.shape[0]:
                depth_value = self.depth_image[cy, cx]
                if not np.isnan(depth_value) and depth_value < min_depth:
                    min_depth = depth_value
                    closest_depth = depth_value

        # Apply angular tolerance
        if angular_error is not None and abs(angular_error) < self.angular_tolerance:
            angular_error = 0.0

        # Debugging log for orientation
        rospy.loginfo(f"Aspect ratio: {aspect_ratio:.2f}, Angular error: {angular_error:.2f}")
        return closest_depth, angular_error

    def maintain_distance(self, leader_depth, angular_error):
        # Time step calculation
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Calculate distance error and derivative
        distance_error = leader_depth - self.desired_distance
        distance_derivative = (distance_error - self.previous_distance_error) / dt if dt > 0 else 0.0
        self.previous_distance_error = distance_error

        # PD control for linear velocity
        linear_velocity = (self.kp_dist * distance_error) + (self.kd_dist * distance_derivative)
        linear_velocity = max(-1.0, min(linear_velocity, 1.0))  # Clamp linear velocity

        # Calculate angular error and derivative
        angle_derivative = (angular_error - self.previous_angle_error) / dt if dt > 0 else 0.0
        self.previous_angle_error = angular_error

        # PD control for angular velocity
        angular_velocity = -((self.kp_angle * angular_error) + (self.kd_angle * angle_derivative))
        angular_velocity = max(-1.0, min(angular_velocity, 1.0))  # Clamp angular velocity

        rospy.loginfo(f"Distance error: {distance_error:.2f}, Linear velocity: {linear_velocity:.2f}")
        rospy.loginfo(f"Angular error: {angular_error:.2f}, Angular velocity: {angular_velocity:.2f}")

        # Publish the calculated velocities
        self.cmd_vel.linear.x = linear_velocity
        self.cmd_vel.angular.z = angular_velocity
        self.pub_cmd_vel.publish(self.cmd_vel)

    def stop_follower(self):
        rospy.loginfo("Leader not detected or invalid depth. Stopping.")
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.pub_cmd_vel.publish(self.cmd_vel)

    def run(self):
        rospy.loginfo("TurtleBot Follower is running...")
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == "__main__":
    try:
        follower = TurtleBotFollower()
        follower.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Follower node terminated.")
