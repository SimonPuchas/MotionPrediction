# Important Things

Right now we use padding to create the batches, as the sequences have variable lengths. This means right now we also train and validate the model on the padded values, which is ultimately not what we want, as it reduces the performance of the model and basically wastes resources. There is the function pack_padded_sequences which allows the model to only train on the actual data, but this transforms the data into a weird form, which i dont know how to use it right now. If you look at the LSTM_testing.py file, you can see in the training loop that i built in some testing to see how the data changes. 

Newest dataset is the lstm_dataset5.pt, here the data is already preprocessed and stored in a pytorch dictionary with X_train, y_train, X_test, y_test, X_val, y_val fields. There is also a testBatching file in the TestScripts folder to test and understand the functionality of the batching and how the data looks afterwards. 

Right now the movements in the dataset have a constant linear_x velocity, later we could add movements with acceleration/decceleration to have more robust data.

# Motion prediction model

The ROS ws is used to generate our own dataset. The data is stored in a tensor(.npy file) containing, time, 3D-position tensor, 3D-orientation tensor and velocity tensor with linear.x, linear.y = 0, and angular.z. Additionally, we store the collected data in .csv files to have something easily human readable to quickly check and understand whats happening. 

## Idea:

We want to train a motion prediction model, which takes the 3D-pose of an object(human, car, box, etc.) and it's velocity(x,y,z), and uses this information to predict the position of the object a few seconds ahead. This should then be used by a robot to dynamically avoid these moving objects by taking according actions, e.g. slowing down, turning left/right around the predicted movement, continuing normally or stopping completely if needed.

This could later be used in Human-Robot interactions, Cobots, delivery robots, etc.