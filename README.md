# Machine learning - Image classification

Today we want to look at different image classification methods with 2 different datasets, MNIST and Fashion MNIST.

### Datasets 

As mentioned in the previous section, we will be performing our test and evaluation on the two MNIST dataset. One is the handwritten digit MNIST and the other is the Fashion MNIST dataset, both dataset has the similar attributes where both datasets contains sets of 28 x 28 greyscale image label under 10 classes.

image example of MNIST with some explaination 

image example of fashion MNIST with some explaination

For more information on the used dataset, you may refer to the links below: <br /> 
[MNIST handwritten digits dataset](yann.lecun.com/exdb/mnist/). <br /> 
[Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). <br /> 

### Methods 

#### Bernoulli Naive Bayes 


#### Logistic regression 
We will use multi-class logistic regression for this task, with softmax function, where the definition can be found below: 

<a href="https://www.codecogs.com/eqnedit.php?latex=$\textbf{x}&space;\in&space;\textbf{R}^{D}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\textbf{x}&space;\in&space;\textbf{R}^{D}$" title="$\textbf{x} \in \textbf{R}^{D}$" /></a>

When there are <b>_K_</b> different classes <b>C<sub>1</sub>,C<sub>2</sub>, ...... , C<sub>k</sub></b>. For each class <b>C<sub>k</sub></b>, we have parameter vector <b>w<sub>k</sub></b> and the posterior probability of the model will be: 

<a href="https://www.codecogs.com/eqnedit.php?latex=h_\textbf{w}(\textbf{x})&space;=&space;p(y&space;=&space;C_k|\textbf{x};\textbf{w})&space;=&space;\frac{\exp(\textbf{w}_k^T\textbf{x})}{\sum_{k=1}^K&space;\exp(\textbf{w}_k^T\textbf{x})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\textbf{w}(\textbf{x})&space;=&space;p(y&space;=&space;C_k|\textbf{x};\textbf{w})&space;=&space;\frac{\exp(\textbf{w}_k^T\textbf{x})}{\sum_{k=1}^K&space;\exp(\textbf{w}_k^T\textbf{x})}" title="h_\textbf{w}(\textbf{x}) = p(y = C_k|\textbf{x};\textbf{w}) = \frac{\exp(\textbf{w}_k^T\textbf{x})}{\sum_{k=1}^K \exp(\textbf{w}_k^T\textbf{x})}" /></a>

#### Convolutional Neural Network 
