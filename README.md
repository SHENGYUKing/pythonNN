# pythonNN
An Implementation of a little Deep Neural Network by Python

## Principle of Deep Neural Network

### activation function
DNN always uses ReLU (Rectified Linear Unit) function as the activation in the middle layers, the formula as follows:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathit{ReLU}\left&space;(&space;Z&space;\right&space;)=\mathit{max}\left\{&space;0,&space;Z&space;\right\}" title="\bg_white \mathit{ReLU}\left ( Z \right )=\mathit{max}\left\{ 0, Z \right\}" />    
Sigmoid function is usually as the activation unit in the last layer of DNN to implement classification:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathit{sigmoid}(z)=\frac{1}{1&plus;\boldsymbol{e}^{-z}}" title="\bg_white \mathit{sigmoid}(z)=\frac{1}{1+\boldsymbol{e}^{-z}}" />    
After input data processed by the linear unit, we can use the above formulas to compute the activation.    
Linear unit just like the following formula:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;Z^{[l]}=W^{[l]}A^{[l-1]}&plus;b^{[l]}" title="\bg_white Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}" />    
In particular, it is noted that the input here is the activation of the previous layer.    

### forward propagate

First, compute the activation of input data using the following formula:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;A^{[l]}=\mathit{g^{[l]}}\left&space;(&space;Z^{[l]}&space;\right&space;)" title="\bg_white A^{[l]}=\mathit{g^{[l]}}\left ( Z^{[l]} \right )" />    
Then, calculate the cost function using cross-entropy formula:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;J=\frac{-1}{m}\left&space;(&space;y\cdot\mathit{log}(AL)^{T}&plus;(1-y)\cdot\mathit{log}(1-AL)^{T}&space;\right&space;)" title="\bg_white J=\frac{-1}{m}\left ( y\cdot\mathit{log}(AL)^{T}+(1-y)\cdot\mathit{log}(1-AL)^{T} \right )" />    
It is not too different from what we did earlier with logistics regression model. This formula is expressed in matrix way, and it actually comes from a statistical concept called maximum likelihood estimation.    

### backward propagate    

In order to reduce the loss, we need to keep updating the parameters until the loss is minimal. So, we should calculate the partial derivatives of the cost function with respect to w and b at first. The backward propagation can help us do that.    
partial derivative of the cost function with respect to W (matrix):    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\frac{\partial&space;J}{\partial&space;W^{[l]}}=\frac{1}{m}\frac{\partial&space;J}{\partial&space;Z^{[l]}}A^{[l-1]T}" title="\bg_white \frac{\partial J}{\partial W^{[l]}}=\frac{1}{m}\frac{\partial J}{\partial Z^{[l]}}A^{[l-1]T}" />    
partial derivative of the cost function with respect to b (matrix):    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\frac{\partial&space;J}{\partial&space;b^{[l]}}=\frac{\partial&space;J}{\partial&space;Z^{[l]}}" title="\bg_white \frac{\partial J}{\partial b^{[l]}}=\frac{\partial J}{\partial Z^{[l]}}" />    
partial derivative of the cost function with respect to A_prev (matrix):    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\frac{\partial&space;J}{\partial&space;A^{[l-1]}}=W^{[l]T}\frac{\partial&space;J}{\partial&space;Z^{[l]}}" title="\bg_white \frac{\partial J}{\partial A^{[l-1]}}=W^{[l]T}\frac{\partial J}{\partial Z^{[l]}}" />    
Where the partial derivative of the cost function with respect to Z is actually the derivative of the activation function.    

### update parameters

Now, we can use the above partial derivatives to update the parameters.    
update weights (w):    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;W=W-\alpha\frac{\partial&space;J}{\partial&space;W}" title="\bg_white W=W-\alpha\frac{\partial J}{\partial W}" />    
update bias (b):    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;b=b-\alpha\frac{\partial&space;J}{\partial&space;b}" title="\bg_white b=b-\alpha\frac{\partial J}{\partial b}" />    
The alpha in the above formulas means learning rate, and it is usually a very small number (about 0.01).    

### predict

At last, take the updated parameters into forward propagation to get the predictions.    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\hat{Y}=\sigma\left&space;(&space;W^{T}X&plus;b&space;\right&space;)" title="\bg_white \hat{Y}=\sigma\left ( W^{T}X+b \right )" />    
In fact that the result y_hat means the probability, and we can design reasonable thresholds for getting the exact conclusion according to reality situation.    

## custom function summary

- generate_dataset(mode, n_samples, noise):    
  generate a virtual dataset to train and test our algorithm    

- sigmoid(Z):    
  compute the sigmoid function of input Z    

- sigmoid_backward(dA, cache):    
  backward propagation of the sigmoid function

- relu(Z):    
  compute the ReLU function of Z, it can perform better in the middle layers of DNN    

- relu_backward(dA, cache):    
  backward propagation for the ReLU function

- nn_init_params(layer_dims):    
  initialize parameters of DNN, specialy, it is used Xavier Initialization    

- linear_forward(A, W, b):    
  compute the linear part of the forward propagation of a layer    

- linear_activation_forward(A_prev, W, b, activation):    
  implement the activation of the layer using specified function    

- nn_forward(X, params):    
  implement the forward propagation of DNN    

- nn_cost(AL, y):    
  compute the cost function of DNN    

- linear_backward(dZ, cache):    
  compute the linear part of the backward propagation of a layer    

- linear_activation_backward(dA, cache, activation):    
  implement the backward propagation of current layer using corresponding function    

- nn_backward(AL, y, caches):    
  implement the backward propagation of DNN    

- update_params(params, grads, learning_rate):    
  update the parameters of DNN layers using gradient descent     

- model(X, y, layers_dims, learning_rate, num_iterations, show):    
  build the DNN model using the functions we defined above    

- predict(X, y, params):    
  implement the prediction of input X using trained model     

- plot_cost(costs, learning_rate):    
  plot the learning curve based on the costs in the whole training process    

- main():    
  main function to run    
