# FocusFlow
Tensorflow JS 3.0 project that determines MPG based on horsepower of a vehicle.
Main takeaways from this project:

The steps in training a machine learning model include:

1) Formulate your task:

  1.1) Is it a regression problem or a classification one?
  
  1.2) Can this be done with supervised learning or unsupervised learning?
  
  1.3) What is the shape of the input data? 
  
  1.3) What should the output data look like?
  
  
 
2) Prepare your data:


  2.1) Clean your data and manually inspect it for patterns when possible
  
  2.2) Shuffle your data before using it for training
  
  2.3) Normalize your data into a reasonable range for the neural network. Usually 0-1 or -1-1 are good ranges for numerical data.
  
  2.4) Convert your data into tensors
  
  

3) Build and run your model:


  3.1) Define your model using tf.sequential or tf.model then add layers to it using tf.layers.*
  
  3.2) Choose an optimizer ( adam is usually a good one), and parameters like batch size and number of epochs.
  
  3.3) Choose an appropriate  loss function for your problem, and an accuracy metric to help your evaluate progress. meanSquaredError is a common loss function for regression problems.
  
  3.4) Monitor training to see whether the loss is going down
  
  
  
4)Evaluate your model


  4.1) Choose an evaluation metric for your model that you can monitor while training. Once it's trained, try making some test predictions to get a sense of prediction quality.
  
