import rospy
import csv
import os
import tf
import numpy as np
from datetime import datetime
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

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
        self.orientation_threshold = 0.01  # rad
        self.position_threshold = 0.005    # m
        
        self.time_data = []        
        self.combined_data = []    
        
        self.setup_run_directory()
        
        try:
            self.csv_file = open(self.csv_path, 'w')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['position', 'rotation', 'velocity'])
            rospy.loginfo("Successfully created CSV file")
        except IOError as e:
            rospy.logerr(f"Failed to create CSV file: {e}")
            raise
        
        self.initial_orientation = None
        self.base_roll = self.base_pitch = self.base_yaw = 0
        
        self.sub_model = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        rospy.loginfo("Subscribed to /gazebo/model_states")

        self.vel_x = self.vel_y = self.vel_z = 0.0
        self.sub_velocity = rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)
        rospy.loginfo("Subscribed to /turtle1/cmd_vel")

        self.shutdown_sub = rospy.Subscriber('/system/shutdown', Bool, self.shutdown_callback)
        rospy.loginfo("Subscribed to /system/shutdown")

    def setup_run_directory(self):
        package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        runs_dir = os.path.join(package_path, 'runs_new')
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)
            rospy.loginfo(f"Created new directory at: {runs_dir}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(runs_dir, f'run_{timestamp}')
        
        os.makedirs(run_dir)
        rospy.loginfo(f"Created run directory at: {run_dir}")
        
        self.run_dir = run_dir
        self.csv_path = os.path.join(run_dir, 'data.csv')
        self.tensor_path = os.path.join(run_dir, 'tensor_data.npy')
        
        rospy.loginfo(f"Data will be saved to: {self.run_dir}")

    def shutdown_callback(self, msg):
        if msg.data:
            rospy.loginfo("Received shutdown signal from robot controller")
            self.shutdown()
            rospy.signal_shutdown("Shutdown signal received")

    def apply_moving_average(self, history, new_value, max_size):
        history.append(new_value)
        if len(history) > max_size:
            history.pop(0)
        return np.mean(history)
    
    def filter_small_changes(self, value, threshold):
        return 0.0 if abs(value) < threshold else value

    def velocity_callback(self, vel_msg):
        self.vel_x = vel_msg.linear.x
        self.vel_y = vel_msg.linear.y
        self.vel_z = vel_msg.angular.z

    def model_states_callback(self, msg):
        try:
            current_time = rospy.Time.now()

            if self.last_time and (current_time - self.last_time).to_sec() < 1:
                return

            idx = msg.name.index('tracked_object')
            current_pose = msg.pose[idx]
            
            quaternion = (
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            )
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
            
            if self.initial_orientation is None:
                self.initial_orientation = quaternion
                self.base_roll, self.base_pitch, self.base_yaw = roll, pitch, yaw
                rospy.loginfo(f"Initial orientation set: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
            
            roll = self.filter_small_changes(roll - self.base_roll, self.orientation_threshold)
            pitch = self.filter_small_changes(pitch - self.base_pitch, self.orientation_threshold)
            yaw = self.filter_small_changes(yaw - self.base_yaw, self.orientation_threshold)
            
            roll = self.apply_moving_average(self.orientation_history, roll, self.orientation_filter_size)
            
            combined = np.array([
                current_pose.position.x,
                current_pose.position.y,
                self.filter_small_changes(current_pose.position.z, self.position_threshold),
                roll,
                pitch,
                yaw,
                self.vel_x,
                self.vel_y,
                self.filter_small_changes(self.vel_z, self.position_threshold)
            ])
            
            elapsed_time = (current_time - self.start_time).to_sec()
            self.last_time = current_time
            
            self.time_data.append(elapsed_time)
            self.combined_data.append(combined)
            
            self.csv_writer.writerow([
                combined[0:3].tolist(),
                combined[3:6].tolist(),
                combined[6:9].tolist()
            ])
            self.csv_file.flush()
            
            self.last_pose = current_pose
            
        except ValueError as e:
            rospy.logwarn(f"tracked_object not found in model states. Available models: {msg.name}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in callback: {e}")
    
    def get_full_tensor(self):
        time_array = np.array(self.time_data)
        combined_array = np.array(self.combined_data)  # Shape: (N, 9)
        
        N = len(self.time_data)
        dtype = [
            #('time', 'f8'),
            ('data', 'f8', (9,))  # Combined position, rotation, and velocity
        ]
        
        full_data = np.zeros(N, dtype=dtype)
        
        full_data['data'] = combined_array
        
        return full_data
    
    def shutdown(self):
        rospy.loginfo("Shutting down pose recorder...")
        if hasattr(self, 'csv_file'):
            try:
                self.csv_file.flush()
                self.csv_file.close()
                rospy.loginfo(f"Successfully closed CSV file: {self.csv_path}")
            
                # Save data if we have any
                if len(self.time_data) > 0:
                    final_tensor = self.get_full_tensor()
                    np.save(self.tensor_path, final_tensor)
                    rospy.loginfo(f"Successfully saved tensor data to {self.tensor_path}")
                else:
                    rospy.logwarn("No data collected, skipping tensor save")
            except Exception as e:
                rospy.logerr(f"Error during shutdown: {e}")

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