#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class TurtleBotFollower:
    def __init__(self):
        rospy.init_node('turtlebot_follower', anonymous=True)

        # Publishers and subscribers
        self.pub_cmd_vel = rospy.Publisher('/follower1/cmd_vel', Twist, queue_size=10)
        self.pub_follower_odom = rospy.Publisher('/follower1/odom', Odometry, queue_size=10)
        self.sub_leader_odom = rospy.Subscriber('/leader/odom', Odometry, self.leader_odom_callback)
        self.sub_follower_odom = rospy.Subscriber('/follower1/odom', Odometry, self.follower_odom_callback)

        # PD control gains
        self.kp_dist = 1  # Increased proportional gain for distance
        self.kd_dist = 0.1 # Increased derivative gain for distance
        self.kp_angle = 1  # Increased proportional gain for angle
        self.kd_angle = 0.1 # Kept derivative gain for angle

        self.desired_distance = 0.707  # Desired distance between follower and leader
        self.previous_distance_error = 0.0
        self.previous_angle_error = 0.0

        self.rate = rospy.Rate(10)  # 10 Hz

        self.leader_odom = None
        self.leader_velocity = 0.0  # Initialize leader velocity
        self.follower_odom = None
        self.cmd_vel = Twist()
        self.last_time = rospy.get_time()

        # Deadband values
        # self.deadband_distance = 0.05  # Distance deadband
        # self.deadband_angle = 0.1  # Angle deadband
        # self.min_linear_velocity = 0.1  # Minimum velocity threshold

    def leader_odom_callback(self, odom):
        self.leader_odom = odom
        self.leader_velocity = odom.twist.twist.linear.x  # Extract the leader's linear velocity
        rospy.loginfo("Leader position: x=%.2f, y=%.2f, velocity: %.2f" % 
                      (odom.pose.pose.position.x, odom.pose.pose.position.y, self.leader_velocity))

    def follower_odom_callback(self, odom):
        self.follower_odom = odom
        rospy.loginfo("Follower position: x=%.2f, y=%.2f" % 
                      (odom.pose.pose.position.x, odom.pose.pose.position.y))

    def wrap_angle(self, angle):
        """ Ensure the angle is within [-pi, pi]. """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_yaw_from_odom(self, odom):
        """ Extract yaw angle from the odometry. """
        orientation = odom.pose.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def update_follower_velocity(self):
        if self.leader_odom is None or self.follower_odom is None:
            rospy.logwarn("Leader or follower odometry not available. Stopping follower.")
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.pub_cmd_vel.publish(self.cmd_vel)
            return

        # Get leader's position and orientation
        leader_x = self.leader_odom.pose.pose.position.x
        leader_y = self.leader_odom.pose.pose.position.y
        leader_yaw = self.get_yaw_from_odom(self.leader_odom)  # Get the leader's yaw angle

        # Calculate target position for the follower with an offset of (-0.5, 0.5)
        offset_x = -0.5 * math.cos(leader_yaw) - 0.5 * math.sin(leader_yaw)  # -0.5 units to the left
        offset_y = -0.5 * math.sin(leader_yaw) + 0.5 * math.cos(leader_yaw)  # 0.5 units up

        target_x = leader_x + offset_x
        target_y = leader_y + offset_y

        # Get follower's current position
        follower_x = self.follower_odom.pose.pose.position.x
        follower_y = self.follower_odom.pose.pose.position.y

        # Calculate distance and angle to the target position
        dx = target_x - follower_x
        dy = target_y - follower_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        current_yaw = self.get_yaw_from_odom(self.follower_odom)
        angle_to_target = self.wrap_angle(target_angle - current_yaw)

        # PD control calculations
        distance_error = distance_to_target - self.desired_distance
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt > 0:
            distance_derivative = (distance_error - self.previous_distance_error) / dt
            angle_derivative = (angle_to_target - self.previous_angle_error) / dt
        else:
            distance_derivative = 0.0
            angle_derivative = 0.0

        self.previous_distance_error = distance_error
        self.previous_angle_error = angle_to_target

        # Apply deadbands
        # if abs(distance_error) < self.deadband_distance:
        #     distance_error = 0.0
        # if abs(angle_to_target) < self.deadband_angle:
        #     angle_to_target = 0.0
        
            # PD control for linear and angular velocity
        linear_velocity = (self.kp_dist * distance_error) + (self.kd_dist * distance_derivative)
        angular_velocity = (self.kp_angle * angle_to_target) + (self.kd_angle * angle_derivative)



        
        # # Ensure the follower has a minimum linear velocity
        # if abs(linear_velocity) < self.min_linear_velocity:
        #     linear_velocity = self.min_linear_velocity * (1 if linear_velocity > 0 else -1)

        # Log calculated velocities
        rospy.loginfo("Linear velocity: %.2f, Angular velocity: %.2f" % (linear_velocity, angular_velocity))

        # Update follower velocity
        self.cmd_vel.linear.x = linear_velocity
        self.cmd_vel.angular.z = angular_velocity
        self.pub_cmd_vel.publish(self.cmd_vel)

        # Publish follower odometry
        self.follower_odom.header.stamp = rospy.Time.now()
        self.pub_follower_odom.publish(self.follower_odom)  # Publish the Odometry message

    def run(self):
        while not rospy.is_shutdown():
            self.update_follower_velocity()
            self.rate.sleep()

if __name__ == '__main__':
    follower = TurtleBotFollower()
    follower.run()
