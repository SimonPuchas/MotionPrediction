import rospy
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool
import math
import tf
import time

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')

        self.waypoints = [
            np.array([0.0, 40.0, 0.0]),
            np.array([0.0, 30.0, 0.0]),
            np.array([0.0, 20.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, -10.0, 0.0]),
        ]
        self.position_tolerance = 0.1
        self.linear_speed = 0.73
        self.angular_speed = 0.72
        
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.state_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)
        self.shutdown_pub = rospy.Publisher('/system/shutdown', Bool, queue_size=1)

        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = 0.0  # Initial yaw angle in radians
        self.current_waypoint_idx = 0
        self.shutdown_initiated = False
        
        rospy.loginfo("Controller initialized with multiple waypoints")

    def state_callback(self, msg):
        try:
            idx = msg.name.index('turtlebot3_waffle')
            self.current_position = np.array([
                msg.pose[idx].position.x,
                msg.pose[idx].position.y,
                msg.pose[idx].position.z
            ])
            orientation_q = msg.pose[idx].orientation
            _, _, self.current_orientation = tf.transformations.euler_from_quaternion([
                orientation_q.x,
                orientation_q.y,
                orientation_q.z,
                orientation_q.w
            ])
            self.control_loop()

        except ValueError:
            rospy.logwarn("turtlebot3_waffle not found in model states.")
    
    def initiate_shutdown(self):
        if not self.shutdown_initiated:
            self.shutdown_initiated = True
            rospy.loginfo("All waypoints reached! Initiating shutdown...")
            rospy.sleep(2.5)
            self.velocity_publisher.publish(Twist())
            self.shutdown_pub.publish(Bool(True))
            rospy.signal_shutdown("All waypoints reached!")

    def control_loop(self):
        if self.current_waypoint_idx < len(self.waypoints):
            target_position = self.waypoints[self.current_waypoint_idx]
            distance_vector = target_position - self.current_position
            distance_to_target = np.linalg.norm(distance_vector)
            target_angle = math.atan2(distance_vector[1], distance_vector[0])
            angle_difference = target_angle - self.current_orientation

            # Normalize the angle difference to be within [-pi, pi]
            angle_difference = math.atan2(math.sin(angle_difference), math.cos(angle_difference))

            if distance_to_target > self.position_tolerance:
                # Create a Twist message for dynamic movement
                vel_msg = Twist()
                vel_msg.linear.x = self.linear_speed  # Constant forward speed
                vel_msg.angular.z = self.angular_speed * angle_difference  # Dynamic angular speed based on angle difference
            
                # Publish the velocity command
                self.velocity_publisher.publish(vel_msg)
            else:
                rospy.loginfo(f"Waypoint {self.current_waypoint_idx + 1} reached.")
                self.current_waypoint_idx += 1
        else:
            self.initiate_shutdown()


if __name__ == '__main__':
    try:
        controller = RobotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
