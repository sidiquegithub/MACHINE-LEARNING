# MACHINE-LEARNING
MACHINE LEARNING ALGORITHMS FROM THE TEXTHANDS ON MACHINE LEARNING WITH SCIKIT LEARN KERAS AND TENSOR FLOW-2nd-Edition-Aurelien-Geron

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
     - Select a more powerful model, with more parameters.
     - Feed better features to the learning algorithm (feature engineering).
     - Reduce the constraints on the model (e.g., reduce the regularization hyperparameter).














