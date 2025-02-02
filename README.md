# Motion prediction model

## Project Structure:

The ROS WS(catkin_ws) is used to generate our own dataset. We let a robot drive around in the Gazebo simulation and record its 6D-pose(x,y,z,roll,pitch,yaw), linear velocity and angular velocity. 

In the Dataset folder we then have the script, which takes the recorded data, performs preprocessing, like standardization, creating sliding windows and splitting into train/val/test sets and then stores it in a pytorch file.

In the ModelScripts folder you can find the scripts, which implement our LSTM model. The LSTM2.py is the currently newest version, has the functionality to store the trained model and uses wandb(weights and biases) to track the training runs. The LSTM_testing.py is used to tune hyperparameters, try new things, etc. There is also the LSTM_evaluation_2.py script, which is used to evaluate the performance of our model on the test set.

The Models folder contains the already trained models. These can be used for testing the performance on the testing dataset and creating a visualization of the output.

In the EvaluationResults folder you can find the results of our trained model, when evaluated on the test set. 

The testingScripts folder contains arbitrary test files. These have no explicit purpose, but rather help to understand what certain methods, scripts, etc. are doing.

The Presentation folder contains all necessary files to run the jupyter notebook correctly. This will be our presentation notebook. 

## How to execute the code:

If you have ROS installed you can use the workspace to generate your own data. The implementation is very basic, so to get different movements you have to manually change the waypoints, linear and angular velocity. All of this is done in the robot_controller.py found here catkin_ws/src/datasetcreator/src/scripts. Then to launch the Gazebo simulation, the robot controller and the recorder you simply have to use the launch file, using the following commands: cd catkin_ws, then source devel/setup.bash, then roslaunch datasetcreator record_motion.launch

The next step would be to run the following file: Datasets/dataset6_creator.py; this will take the collected data, perform normalization, cutting all movements to the same length, applying sliding windows, splitting it into train/val/test and storing it as a pytorch dictionary.

If you don't have the possibility to create your own data you can simply use the already available dataset in the Datasets folder. 

Then you can go to the ModelScripts and execute LSTM_2.py, this script will train the model and store it in the Models folder.

Once you trained the Model and saved it, you can proceed to the evaluation. For this you run the following file LSTM_evaluation_2.py. The results are stored in the EvaluationResults.

## Idea:

We want to train a motion prediction model, which takes the 6D-pose of an object(in our case a robot) and it's velocity(linear, angular), and uses this information to predict the position of the object a few seconds ahead.
So the model observes, e.g. the last 10 seconds of a movement and creates a prediction based on this time window. The prediction is one timestep, so 1 second, which is the timestep that comes after the last 10 seconds.

Example:
| x | y | z | roll | pitch | yaw | linear vel | angular vel |
|-------|-------|-------|-------|-------|-------|-------|-------|
| 0  | 0  | 0  | 0  | 0  | 0  | 1  | 0.5  |
| 1  | 1  | 0  | 0  | 0  | 0.2  | 1  | 0.5  |
| 2  | -1  | 0  | 0  | 0  | -0.2  | 1  | 0.5  |
| 3  | 1  | 0  | 0  | 0  | 0.2  | 1  | 0.5  |
| 4  | -1  | 0  | 0  | 0  | -0.2  | 1  | 0.5  |
| 5  | 1  | 0  | 0  | 0  | 0.2  | 1  | 0.5  |
| 6  | -1  | 0  | 0  | 0  | -0.2  | 1  | 0.5  |
| 7  | 1  | 0  | 0  | 0  | 0.2  | 1  | 0.5  |
| 8  | -1  | 0  | 0  | 0  | -0.2  | 1  | 0.5  |
| 9  | 1 | 0 | 0 | 0 | 0.2 | 1 | 0.5 |
| 10 | -1 | 0 | 0 | 0 | -0.2 | 1 | 0.5 |

Prediction should look like the last row from above:
| x | y | z | roll | pitch | yaw | linear vel | angular vel |
|-------|-------|-------|-------|-------|-------|-------|-------|
| 9.985 | -1.025 | 0 | -0.052 | 0.123 | -0.19346 | 1.0104 | 0.4245 |

But the result will slightly deviate from the actual values since the model wont learn the perfect values. 

This can then be used by a robot to dynamically avoid these moving objects by taking according actions, e.g. slowing down, turning left/right around the predicted movement, continuing normally or stopping completely if needed. Such a model could be used for dynamic real-time obstacle avoidance in a dynamic world, where objects are not static. 
