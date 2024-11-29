# Important Things

Rather than using padding in the dataset creation to get the sequences to the same length, we should use packing functions in the LSTM training loop.
We will also apply normalization directly in the training script to keep flexibility.

Newest dataset is the lstm_dataset5.pt, here the data is already preprocessed and stored in a pytorch dictionary with X_train, y_train, X_test, y_test, X_val, y_val fields. There is also a testBatching file in the TestScripts folder to test and understand the functionality of the batching and how the data looks afterwards. 

Right now the movements in the dataset have a constant linear_x velocity, later we could add movements with acceleration/decceleration to have more robust data.

# Motion prediction model

The ROS ws is used to generate our own dataset. The data is stored in a tensor(.npy file) containing, time, 3D-position tensor, 3D-orientation tensor and velocity tensor with linear.x, linear.y = 0, and angular.z. Additionally, we store the collected data in .csv files to have something easily human readable to quickly check and understand whats happening. 

## Idea:

We want to train a motion prediction model, which takes the 3D-pose of an object(human, car, box, etc.) and it's velocity(x,y,z), and uses this information to predict the position of the object a few seconds ahead. This should then be used by a robot to dynamically avoid these moving objects by taking according actions, e.g. slowing down, turning left/right around the predicted movement, continuing normally or stopping completely if needed.

This could later be used in Human-Robot interactions, Cobots, delivery robots, etc.

## Project Structure

### Phase 1: (until 8.11 to 10.11)

* Collect Datasets; generate own data; create one clean dataset;
* Do research; look for usable architecture; search for similar projects and get some inspiration; collect helpful info
* Set a goal(how sophisticated/complex should the model be); split work, such that everyone has a roughly defined workflow

### Phase 2: (as soon as Phase 1 is done)

* Create a basic model, able to predict simple movements and make sure it works very well 
* high accuracy very important, as you don't want to have collisions
* Generate more specific data, which is especially useful for this usecase

### Phase 3: (basic setup can start shortly after Phase 2, implementation once model is good enough)

* Start setting up ROS environment; programming basic navigation, obstacle avoidance
* Then implement basic motion prediction model

### Phase 4: (depending on goal and time left)
* Make model more complex, e.g. boxes falling from above, close encounters, changing directions, etc.
* Implement new model in ROS and create more complex environment