#### Instructions
You're going to build a model on the digits dataset, a sample dataset that comes pre-loaded with scikit learn. 
The digits dataset consist of 8x8 pixel handwritten digits from 0 to 9:
You want to distinguish between each of the 10 possible digits given an image, so we are dealing with multi-class classification.
The dataset has already been partitioned into X_train, y_train, X_test, and y_test, using 30% of the data as testing data. 
The labels are already one-hot encoded vectors, so you don't need to use Keras to_categorical() function.
Let's build this new model!

---
#### Instructions
Add a Dense layer of 16 neurons with relu activation and an input_shape that takes the total number of pixels of the 8x8 digit image.
Add a Dense layer with 10 outputs and softmax activation.
Compile your model with adam, categorical_crossentropy, and accuracy metrics.
Make sure your model works by predicting on X_train.

---

```
# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape = (64,), activation = 'relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation = 'softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))
```
---
#### Is the model overfitting?
Let's train the model you just built and plot its learning curve to check out if it's overfitting! You can make use of the loaded function plot_loss() to plot training loss against validation loss, you can get both from the history callback.
If you want to inspect the plot_loss() function code, paste this in the console: show_code(plot_loss)
* Train your model for 60 epochs, using X_test and y_test as validation data.
* Use plot_loss() passing loss and val_loss as extracted from the history attribute of the h_callback object.

---
```
# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])
```

---
#### Do we need more data?
It's time to check whether the digits dataset model you built benefits from more training examples!

In order to keep code to a minimum, various things are already initialized and ready to use:

The model you just built.
X_train,y_train,X_test, and y_test.
The initial_weights of your model, saved after using model.get_weights().
A pre-defined list of training sizes: training_sizes.
A pre-defined early stopping callback monitoring loss: early_stop.
Two empty lists to store the evaluation results: train_accs and test_accs.
Train your model on the different training sizes and evaluate the results on X_test. End by plotting the results with plot_results().

The full code for this exercise can be found on the slides!

#### Instructions

* Get a fraction of the training data determined by the size we are currently evaluating in the loop.
* Set the model weights to the initial_weights with set_weights() and train your model on the fraction of training data using early_stop as a callback.
* Evaluate and store the accuracy for the training fraction and the test set.
* Call plot_results() passing in the training and test accuracies for each training size.
---
```
for size in training_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new training data fraction
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [early_stop])

    # Evaluate and store both: the training data fraction and the complete test set results
    train_accs.append(model.evaluate(X_train_frac, y_train_frac)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])
    
# Plot train vs test accuracies
plot_results(train_accs, test_accs)
```
