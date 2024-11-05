import rospy
import csv
import os
import tf
import numpy as np
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist

class DataRecorder:
    def __init__(self):
        rospy.init_node('data_recorder')
        self.start_time = rospy.Time.now()
        self.last_pose = None
        self.last_time = None
        
        # Filter parameters
        self.orientation_filter_size = 5
        self.orientation_history = []
        
        # Thresholds for noise filtering
        self.orientation_threshold = 0.01  # rad - orientation changes below this are considered noise
        self.position_threshold = 0.005     # m - position changes below this are considered noise, only used for z-axis noise
        
        # Get package path and create full file path
        package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.csv_path = os.path.join(package_path, 'data.csv')
        rospy.loginfo(f"Writing data to: {self.csv_path}")
        
        # Create CSV file
        try:
            self.csv_file = open(self.csv_path, 'w')    # w for overwriting, a for appending
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vel_x', 'vel_y', 'vel_z'])
            #self.csv_writer.writerow(['new', 'movement'])
            rospy.loginfo("Successfully created CSV file")
        except IOError as e:
            rospy.logerr(f"Failed to create CSV file: {e}")
            raise
        
        # Initial orientation values
        self.initial_orientation = None
        self.base_roll = self.base_pitch = self.base_yaw = 0
        
        # Subscribe to model states
        self.sub_model = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        rospy.loginfo("Subscribed to /gazebo/model_states")

        # Subscribe to velocity commands
        self.vel_x = self.vel_y = self.vel_z = 0.0
        self.sub_velocity = rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)
        rospy.loginfo("Subscribed to /turtle1/cmd_vel")

    def apply_moving_average(self, history, new_value, max_size):
        """Apply moving average filter to a value"""
        history.append(new_value)
        if len(history) > max_size:
            history.pop(0)
        return np.mean(history)
    
    def filter_small_changes(self, value, threshold):
        """Zero out values below threshold"""
        return 0.0 if abs(value) < threshold else value

    def velocity_callback(self, vel_msg):
        """Callback function to extract velocities from the Twist message."""
        self.vel_x = vel_msg.linear.x
        self.vel_y = vel_msg.linear.y
        self.vel_z = vel_msg.linear.z

    def model_states_callback(self, msg):
        try:
            idx = msg.name.index('tracked_object')
            current_pose = msg.pose[idx]
            current_time = rospy.Time.now()
            
            # Get orientation
            quaternion = (
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            )
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
            
            # Store initial orientation as reference
            if self.initial_orientation is None:
                self.initial_orientation = quaternion
                self.base_roll, self.base_pitch, self.base_yaw = roll, pitch, yaw
                rospy.loginfo(f"Initial orientation set: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
            
            # Calculate relative orientation to initial position
            roll = self.filter_small_changes(roll - self.base_roll, self.orientation_threshold)
            pitch = self.filter_small_changes(pitch - self.base_pitch, self.orientation_threshold)
            yaw = self.filter_small_changes(yaw - self.base_yaw, self.orientation_threshold)
            
            # Apply orientation filtering
            roll = self.apply_moving_average(self.orientation_history, roll, self.orientation_filter_size)
            pitch = pitch  # Keep pitch as is since it should be constant
            yaw = yaw     # Keep yaw as is since it should be constant

            x_pos = current_pose.position.x
            y_pos = current_pose.position.y
            # We want to filter out z-axis noise, as we only have horizontal movement
            z_pos = self.filter_small_changes(current_pose.position.z, self.position_threshold)
            
            # Record data
            # We might want to combine the 3d pos, r,p,y and 3d velocity each into single features using tensors or similar
            elapsed_time = (current_time - self.start_time).to_sec()
            row_data = [
                elapsed_time,
                x_pos,
                y_pos,
                z_pos,
                roll,   # Might omit this feature, if we only use horizontal movement
                pitch,  # Might omit this feature, if we only use horizontal movement
                yaw,    # This changes when turning
                self.vel_x,  
                self.vel_y,
                self.vel_z
            ]
            
            self.csv_writer.writerow(row_data)
            self.csv_file.flush()
            
            self.last_pose = current_pose
            self.last_time = current_time
            
        except ValueError as e:
            rospy.logwarn(f"tracked_object not found in model states. Available models: {msg.name}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in callback: {e}")
    
    def shutdown(self):
        rospy.loginfo("Shutting down pose recorder...")
        if hasattr(self, 'csv_file'):
            try:
                self.csv_file.flush()
                self.csv_file.close()
                rospy.loginfo(f"Successfully closed CSV file: {self.csv_path}")
            except Exception as e:
                rospy.logerr(f"Error closing CSV file: {e}")

if __name__ == '__main__':
    try:
        recorder = DataRecorder()
        rospy.on_shutdown(recorder.shutdown)
        rospy.loginfo("Pose recorder node is running. Press Ctrl+C to stop...")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Failed to start pose recorder: {e}")
