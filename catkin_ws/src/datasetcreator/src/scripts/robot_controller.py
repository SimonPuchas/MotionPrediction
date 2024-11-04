import rospy
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
import tf
import os

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')

        # Parameters
        self.target_position = np.array([10.0, 0.0, 0.0])
        self.position_tolerance = 0.1
        self.linear_speed = 0.5
        
        # Initialize publisher and subscriber
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.state_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)

        # Current position
        self.current_position = np.array([0.0, 0.0, 0.0])
        
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
            # Stop when target is reached
            self.velocity_publisher.publish(Twist())
            #rospy.loginfo("Target reached!")
            rospy.signal_shutdown("Target reached!")
            os.system("rosnode kill -a")
            os.system("killall -9 gzserver")
            os.system("killall -9 gzclient")


if __name__ == '__main__':
    try:
        controller = RobotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass