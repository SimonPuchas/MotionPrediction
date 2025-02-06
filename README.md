# Motion prediction model

## Report

All the details about this project are found in the AMPMReport.pdf file. This report delves into the theoretical aspects of the project. Instead, the technical implementation can be found in the Presentation/Presentation.ipynb Jupyter Notebook.

The model was ran several times with Wandb (Weights & Biases). You can see the overall picture of the runs in AMPMVal_Loss.png

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

Alternatively, you can just run the Presentation.ipynb notebook. The only things that have to be changed are the input and output folder paths with your own folder path.


