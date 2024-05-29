# MACHINE-LEARNING
MACHINE LEARNING ALGORITHMS FROM THE TEXTHANDS ON MACHINE LEARNING WITH SCIKIT LEARN KERAS AND TENSOR FLOW-2nd-Edition-Aurelien-Geron

CHAPTER 1 FUNDAMENTALS OF MACHINE LEARNING
- What is machine learning
- Why Use Machine Learning?
- Types of Machine Learning Systems
  - Supervised/Unsupervised Learning
  - Batch and Online Learning
- Main Challenges of Machine Learning
  - Examples of Bad Data
  - A Couple of Examples of Bad Algorithms.
    - Overfitting
    - Underfitting
- Testing and Validating
- Hyperparameter Tuning and Model Selection
  - Holdout validation
  - Cross-validation


# CHAPTER 1 FUNDAMENTALS OF MACHINE LEARNING

## What is machine learning
- Machine Learning is the science (and art) of programming computers so they can learn from data.

- Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. —Arthur Samuel, 1959
- A computer program is said to learn from experience E with respect to some task Tand some performance measure P, if its performance on T, as measured by P,improves with experience E.
     —Tom Mitchell, 1997
  
## Why Use Machine Learning?

- Problems for which existing solutions require a lot of fine-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better than the traditional approach.
- Complex problems for which using a traditional approach yields no good solution: the best Machine Learning techniques can perhaps find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.

## Types of Machine Learning Systems
- Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning)
-  Whether or not they can learn incrementally on the fly (online versus batch learning)
-  Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning)

These criteria are not exclusive; you can combine them in any way you like. For
example, a state-of-the-art spam filter may learn on the fly using a deep neural net‐
work model trained using examples of spam and ham; this makes it an online, modelbased, supervised learning system

### Supervised/Unsupervised Learning
Machine Learning systems can be classified according to the amount and type of supervision they get during training. 
There are four major categories: 
  - Supervised learning.
  - Unsupervised learning
  - Semisupervised learning.
  - Reinforcement learning.

Supervised learning

- In supervised learning, the training set you feed to the algorithm includes the desired solutions, called labels.
- A typical supervised learning task is classification.
  - The spam filter is a good example of this: it is trained with many example emails along with their class (spam or ham),and it must learn how to classify new emails.
- Another supervised learning task is Regression
  - To predict a target numeric value, such as the price of a car, given a set of features (mileage, age, brand, etc.) called predictors.
- Note that some regression algorithms can be used for classification as well, and vice versa. For example, Logistic Regression is commonly used for classification, as it can
output a value that corresponds to the probability of belonging to a given class

Unsupervised learning
 - In unsupervised learning, as you might guess, the training data is unlabeled. The system tries to learn without a teacher.
     - For example, say you have a lot of data about your blog’s visitors. You may want to run a clustering algorithm to try to detect groups of similar visitors.


NOTE:$Dimensionality \ Reduction$ ; The goal is to simplify the datawithout losing too much information.
     - One way to do this is to merge several correla‐ted features into one. This is called $feature \ extraction$.  

Semisupervised learning

- Some algorithms can deal with data that’s partially labeled. This is called semisupervised learning.
- Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just add one label per person4 and it is able to name everyone in every photo, which is useful for searching photos.

Reinforcement Learning
- Reinforcement Learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get
rewards in return (or penalties in the form of negative rewards)

### Batch and Online Learning
Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.

Batch learning
- In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data.
- This will generally take a lot of time and computing resources, so it is typically done offline.
-  First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called $offline \ learning$.

Online learning

- In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning
step is fast and cheap, so the system can learn about new data on the fly, as it arrives.
- Online learning is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously.
- One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. If you set a high learning rate, then your
system will rapidly adapt to new data, but it will also tend to quickly forget the old data.
Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to
sequences of nonrepresentative data points (outliers).

### Main Challenges of Machine Learning

The two things that can go wrong are “bad algorithm” and “bad data".

Examples of Bad Data
- Insufficient Quantity of Training Data
- Nonrepresentative Training Data
- Poor-Quality Data
- Irrelevant Features

A Couple of Examples of Bad Algorithms.

- Overfitting
- Underfitting the Training Data

     - _Overfitting_: It tmeans that the model performs well on the training data, but it does not generalizewell.
     - Overfitting happens when the model is too complex relative to the amount and noisiness of the training data.
     - Here are possible solutions:
          - Simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomialmodel), by reducing the number of attributes in the training
          data, or by constraining the model.
          - Gather more training data.
          - Reduce the noise in the training data (e.g., fix data errors and remove outliers).
 
     - Constraining a model to make it simpler and reduce the risk of overfitting is called $regularization$.
     - The amount of regularization to apply during learning can be controlled by a $hyperparameter$. A hyperparameter is a parameter of a learning algorithm (not of the
model). As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training.


     - _Underfitting the Training Data_
     - It occurs when your model is too simple to learn the underlying structure of the data. For example, a linear model of life satisfaction is prone to underfit; reality is just more complex than
the model, so its predictions are bound to be inaccurate, even on the training
examples
     - Select a more powerful model, with more parameters.
     - Feed better features to the learning algorithm (feature engineering).
     - Reduce the constraints on the model (e.g., reduce the regularization hyperparameter).



### Testing and Validating
The only way to know how well a model will generalize to new cases is to actually try
it out on new cases.

One option is to split your data into two sets: the training set and the test set. As these names imply, you train your model using the training set, and you test it using
the test set. The error rate on new cases is called the $generalization \ error \ (or \ out-of \ sample \ error),$ and by evaluating your model on the test set, you get an estimate of this
error. This value tells you how well your model will perform on instances it has never seen before.

If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is overfitting the train
ing data.

### Hyperparameter Tuning and Model Selection

$Holdout validation$
Simply hold out part of the training set to evaluate several candidate models and select the best one. 

More specifically, you train multiple models with various hyperparameters on the reduced training set (i.e., the full training set minus the validation set), and
you select the model that performs best on the validation set. After this holdout validation process, you train the best model on the full training set (including the validation set), and this gives you the final model. Lastly, you evaluate this final model on the test set to get an estimate of the generalization error

Drawbacks of holdout validation : if the validation set is too small, then model evaluations will be imprecise: you may end up selecting a suboptimal model by mistake. Conversely, if the validation set is too large, then the remaining training set will be much smaller than the full training set. 


 $Cross-validation$ 
 Each model is evaluated once per validation set after it is trained on the rest of the data. By averaging out all Each model is evaluated
once per validation set after it is trained on the rest of the data. By averaging out all. 

 

# CHAPTER 1 END TO END MACHINE LEARNING PROJECT

Main Steps

1. Look at the big picture.
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data for Machine Learning algorithms.
5. Select a model and train it.
6. Fine-tune your model.
7. Present your solution.
8. Launch, monitor, and maintain your system.

## Look at the big picture
### Frame the Problem
The first question to ask your boss is what exactly the business objective is. Building a
model is probably not the end goal. How does the company expect to use and benefit
from this model? Knowing the objective is important because it will determine how
you frame the problem, which algorithms you will select, which performance meas‐
ure you will use to evaluate your model, and how much effort you will spend tweak‐
ing it.

First, you need to frame the problem: is it supervised, unsupervised, or Reinforcement
Learning? Is it a classification task, a regression task, or something else? Should you
use batch learning or online learning techniques?

### Select a performance measure
A typical performance measure for regression problems is the Root Mean Square Error (RMSE).
$$RMSE = \sqrt{\frac{1}{m} \sum {(h(x_i)  - y_i)}^2 }$$

Even though the RMSE is generally the preferred performance measure for regression
tasks, in some contexts you may prefer to use another function. For example, suppose
that there are many outlier districts. In that case, you may consider using the mean
absolute error (MAE, also called the average absolute deviation) 

$$ MAE = \frac {1}{m} \sum |h(x_i) - y_i|$$

The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE

### Download the data
### Take a quick look at the data
Use functions like .head() , .hist()  etc
### Create a test data
When you estimate the generalization error using the test set, your estimate will be too optimistic, and you will launch a system that will not
perform as well as expected. This is called $data\ snooping \ bias$.

Scikit-Learn provides a few functions to split datasets into multiple subsets in various
ways. The simplest function is train_test_split(). First, there is a random_state parameter that allows you to set the random generator
seed. Second, you can pass it multiple datasets with an identical number of rows, and
it will split them on the same indices (this is very useful, for example, if you have a
separate DataFrame for labels):

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

So far we have considered purely random sampling methods. This is generally fine if
your dataset is large enough (especially relative to the number of attributes), but if it
is not, you run the risk of introducing a significant $sampling\ bias$. 

When a survey company decides to call 1,000 people to ask them a few questions, they don’t just pick
1,000 people randomly in a phone book. They try to ensure that these 1,000 people
are representative of the whole population. For example, the US population is 51.3%
females and 48.7% males, so a well-conducted survey in the US would try to maintain
this ratio in the sample: 513 female and 487 male. This is called $stratified \ sampling $:
the population is divided into homogeneous subgroups called strata, and the right
number of instances are sampled from each stratum to guarantee that the test set is
representative of the overall population.

We can use Scikit-Learn’s StratifiedShuffleSplit class:

### Discover and Visualize the Data to Gain Insights
