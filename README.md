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


NOTE: Dimensionality Reduction ; The goal is to simplify the datawithout losing too much information.
     - One way to do this is to merge several correla‐ted features into one. This is called feature extraction.  















