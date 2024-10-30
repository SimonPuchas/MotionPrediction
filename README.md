# Project structure:

## Idea:

We want to train a motion prediction model, which takes the 3D-pose of an object(human, car, box, etc.) and it's velocity(x,y,z), and uses this information to predict the position of the object a few seconds ahead. This should then be used by a robot to dynamically avoid these moving objects by taking according actions, e.g. slowing down, turning left/right around the predicted movement, continuing normally or stopping completely if needed.