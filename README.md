# Machine learning - Image classification

Today we want to look at different image classification methods with 2 different datasets, MNIST and Fashion MNIST.

### Datasets 

As mentioned in the previous section, we will be performing our test and evaluation on the two MNIST dataset. One is the handwritten digit MNIST and the other is the Fashion MNIST dataset, both dataset has the similar attributes where both datasets contains sets of 28 x 28 greyscale image label under 10 classes.

<p><a href="https://commons.wikimedia.org/wiki/File:MnistExamples.png#/media/File:MnistExamples.png"><img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" alt="MNIST sample images"></a><br>By <a href="//commons.wikimedia.org/w/index.php?title=User:Jost_swd15&amp;action=edit&amp;redlink=1" class="new" title="User:Jost swd15 (page does not exist)">Josef Steppan</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=64810040">Link</a></p>

Fashion Mnist dataset is created because hand written digits MNIST dataset is too simple, where it can easily achieve 90% accuracy with simple model setup.
![Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png)
Image taken from Fashion-MNIST oficial github website.

For more information on the used dataset, you may refer to the links below: <br /> 
[MNIST handwritten digits dataset](yann.lecun.com/exdb/mnist/). <br /> 
[Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). <br /> 

### Methods 

#### Bernoulli Naive Bayes 
Naive Bayes is a probabilitic classifier that has strong (naïve) independence assumptions between the features. Out of all the Naive Bayes model, we chose to work with bernoulli naive bayes as I believe that we can treat the image data as binary representation when calculating the posterior hence it should work relatively better compared to Gaussian or Multinomial Naive Bayes model.

The definition of the likelihood of each Class <b>C<sub>k</sub></b> is shown below: 
<a href="https://www.codecogs.com/eqnedit.php?latex=p(x|C_k)&space;=&space;\prod_{i=1}^{n}&space;p_{k_i}^{x_i}&space;(1&space;-&space;p_{k_i})^{(1-x_i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x|C_k)&space;=&space;\prod_{i=1}^{n}&space;p_{k_i}^{x_i}&space;(1&space;-&space;p_{k_i})^{(1-x_i)}" title="p(x|C_k) = \prod_{i=1}^{n} p_{k_i}^{x_i} (1 - p_{k_i})^{(1-x_i)}" /></a>

Additional reading material: <br />
[Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes)
[NLP Standford](https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html)

#### Logistic regression 
We will use multi-class logistic regression for this task, with softmax function, where the definition can be found below: 

<a href="https://www.codecogs.com/eqnedit.php?latex=$\textbf{x}&space;\in&space;\textbf{R}^{D}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\textbf{x}&space;\in&space;\textbf{R}^{D}$" title="$\textbf{x} \in \textbf{R}^{D}$" /></a>

When there are <b>_K_</b> different classes <b>C<sub>1</sub>,C<sub>2</sub>, ...... , C<sub>k</sub></b>. For each class <b>C<sub>k</sub></b>, we have parameter vector <b>w<sub>k</sub></b> and the posterior probability of the model will be: 

<a href="https://www.codecogs.com/eqnedit.php?latex=h_\textbf{w}(\textbf{x})&space;=&space;p(y&space;=&space;C_k|\textbf{x};\textbf{w})&space;=&space;\frac{\exp(\textbf{w}_k^T\textbf{x})}{\sum_{k=1}^K&space;\exp(\textbf{w}_k^T\textbf{x})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\textbf{w}(\textbf{x})&space;=&space;p(y&space;=&space;C_k|\textbf{x};\textbf{w})&space;=&space;\frac{\exp(\textbf{w}_k^T\textbf{x})}{\sum_{k=1}^K&space;\exp(\textbf{w}_k^T\textbf{x})}" title="h_\textbf{w}(\textbf{x}) = p(y = C_k|\textbf{x};\textbf{w}) = \frac{\exp(\textbf{w}_k^T\textbf{x})}{\sum_{k=1}^K \exp(\textbf{w}_k^T\textbf{x})}" /></a>

Where the formula of multinomial logistics loss is:

<a href="https://www.codecogs.com/eqnedit.php?latex=\textbf{J}(\textbf{w})&space;=&space;-&space;\left[\sum_{n=1}^{N}&space;\sum_{k=1}^K&space;1_{\{y^{n}&space;=&space;C_k\}}log(h_\textbf{w}(\textbf{x}^n))\right]&space;=&space;-&space;\left[\sum_{n=1}^{N}&space;\sum_{k=1}^K&space;1_{\{y^{n}&space;=&space;C_k\}}log(y^{n}&space;=C_k|&space;\textbf{x}^{n};\textbf{w})\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf{J}(\textbf{w})&space;=&space;-&space;\left[\sum_{n=1}^{N}&space;\sum_{k=1}^K&space;1_{\{y^{n}&space;=&space;C_k\}}log(h_\textbf{w}(\textbf{x}^n))\right]&space;=&space;-&space;\left[\sum_{n=1}^{N}&space;\sum_{k=1}^K&space;1_{\{y^{n}&space;=&space;C_k\}}log(y^{n}&space;=C_k|&space;\textbf{x}^{n};\textbf{w})\right]" title="\textbf{J}(\textbf{w}) = - \left[\sum_{n=1}^{N} \sum_{k=1}^K 1_{\{y^{n} = C_k\}}log(h_\textbf{w}(\textbf{x}^n))\right] = - \left[\sum_{n=1}^{N} \sum_{k=1}^K 1_{\{y^{n} = C_k\}}log(y^{n} =C_k| \textbf{x}^{n};\textbf{w})\right]" /></a>

Additional reading material:  <br />
[Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression) <br />
[Stackoverflow](https://stackoverflow.com/questions/36051506/difference-between-logistic-regression-and-softmax-regression) <br />

#### Convolutional Neural Network 

Anyone that has some exposure to machine learning would have heard of convolutional neural network, it is one of the most commonly used machine learning method for analyzing visual imagery. 

Additional reading material: <br />
[Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network) <br />
[Adit Deshpande](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) <br />

### Results 

Without any doubts we can directly assume that Neural Network will have better results compared to the other machine learning methods and the final results shows. However, Bernoulli Naive Bayes Classifier is getting comparable results to logistic regression which is kinda of a surprise where Naive Bayes does not really have good prediction results in real life tasks though that work really well in theory. But our current implementation of logistic regression is not optimal where if we have implement regularization we can sure to see huge improvement in creating a more robust classifier. Though CNN have achieved a better prediction results, there are extremely sensitive to different dataset, meaning with different task you will need to define different architechture where there is no such thing as one model to fit all (No free lunch theory).

From the table below, You can find the summary of accuracy/performance of all the different classifiers for MNIST and Fashion-MNIST dataset.
![Result](https://github.com/MingSheng92/Image_Classification/blob/master/img/result.JPG)

### Future Work 
1. Add regularization to logistic regression to add robustness to the classifier.
2. Implement multinomial Naive Bayes and Gaussian Naive Bayes and compare between the Naive Bayes models.
3. Add different evaluation methods to further analyze the listed models, ROC curve, confusion matrix etc.
