import rospy
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool
import tf
import os
import subprocess

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')

        # Parameters
        self.target_position = np.array([10.0, 0.0, 0.0])
        self.position_tolerance = 0.001
        self.linear_speed = 0.5
        
        # Initialize publisher and subscriber
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.state_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)
        
        # Add shutdown publisher
        self.shutdown_pub = rospy.Publisher('/system/shutdown', Bool, queue_size=1)

        # Current position
        self.current_position = np.array([0.0, 0.0, 0.0])
        
        # Flag to prevent multiple shutdown signals
        self.shutdown_initiated = False
        
        rospy.loginfo("Controller initialized - moving to x=10")

    def state_callback(self, msg):
        try:
            idx = msg.name.index('tracked_object')
            self.current_position = np.array([
                msg.pose[idx].position.x,
                msg.pose[idx].position.y,
                msg.pose[idx].position.z
            ])

            # Print current position every second
            #rospy.loginfo(f"Current position: x={self.current_position[0]:.2f}")
            
            self.control_loop()

        except ValueError:
            rospy.logwarn("tracked_object not found in model states.")
    
    def initiate_shutdown(self):
        """Handle the shutdown sequence"""
        if not self.shutdown_initiated:
            self.shutdown_initiated = True
            rospy.loginfo("Target reached! Initiating shutdown sequence...")
            
            # Stop the robot
            self.velocity_publisher.publish(Twist())
            
            # Publish shutdown signal for other nodes
            self.shutdown_pub.publish(Bool(True))
            
            # Wait briefly to ensure message is published
            rospy.sleep(1.0)
            
            # Shutdown Gazebo
            try:
                subprocess.call(['killall', 'gzserver'])
                subprocess.call(['killall', 'gzclient'])
            except Exception as e:
                rospy.logerr(f"Error shutting down Gazebo: {e}")
            
            # Shutdown ROS
            rospy.signal_shutdown("Target reached!")
    
    def control_loop(self):
        """Simple control loop - just move forward until x=10"""
        distance_to_target = self.target_position[0] - self.current_position[0]
        
        if abs(distance_to_target) > self.position_tolerance:
            # Create velocity command - just move in x direction
            vel_msg = Twist()
            vel_msg.linear.x = self.linear_speed if distance_to_target > 0 else -self.linear_speed
            
            # Publish velocity command
            self.velocity_publisher.publish(vel_msg)
        else:
            # Target reached, initiate shutdown sequence
            self.initiate_shutdown()


if __name__ == '__main__':
    try:
        controller = RobotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass