import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
import os

def move():
    rospy.init_node('path_follower')
    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    speed_x = 1
    speed_y = 0
    speed_z = 0

    rate = rospy.Rate(10)  # 10 Hz
    
    rospy.wait_for_service('gazebo/set_model_state')
    set_state = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

    start = np.array([0, 0, 0])
    end = np.array([10, 0, 0])

    model_state = ModelState()
    model_state.model_name = 'tracked_object'
    model_state.reference_frame = 'world'

    while not rospy.is_shutdown():
        # Set the position of the object
        model_state.pose.position.x = start[0]
        model_state.pose.position.y = start[1]
        model_state.pose.position.z = start[2]

        try:
            set_state(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

        # Set the velocity of the object
        vel_msg.linear.x = speed_x
        vel_msg.linear.y = speed_y
        vel_msg.linear.z = speed_z
        velocity_publisher.publish(vel_msg)

        # Wait for the object to reach the end position
        while not rospy.is_shutdown() and np.linalg.norm(end - np.array([model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z])) > 0.1:
            try:
                # Update the model state to get the latest position
                model_state = set_state(model_state)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
            rate.sleep()
        
        # Stop the object
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        velocity_publisher.publish(vel_msg)

        # Exit after moving once
        
        break  # Remove this if you want continuous movement
    rospy.loginfo("End point reached, shutting down nodes.")
    rospy.signal_shutdown("Path follower has reached the end point")
    os.system("rosnode kill -a")
    os.system("killall -9 gzserver")
    os.system("killall -9 gzclient")

if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass
