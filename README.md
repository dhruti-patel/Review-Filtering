# Review-Filtering
Part 3: Truth be Told

Formulation of the Problem:
We have used Naive Bayes Classifier to solve this problem. We are calculating odds ratio for 2 classes "truthful" and "deceptive". Odds Ratio was calculated by calculating the likelihood probablities of each word in the review for each classes using the training data set. If the odds ratio is greater than 1 we assigned the review a truthful label or else a deceptive label. We were able to get the classification accuracy of 83.75% for provided test data. Also our training data accuracy is 89%.

Description of how the problem works:
Initially we started with getting rid of the noise from the training data so that the cleaned data can be then used to train our model. We eliminated bunch of stop words from the training data as they don't provide any insight on the truthfulness of the review. Then we removed punctuations and special characters if any. We are storing the count of each word in the training data set in 2 seperate dictionaries. One for truthful reviews and other for deceptive reviews. We also ignored the words that occur less than 25 times in the training data set. We found this threshold from trail and error and found this to work best for accuracy.
After getting our training data cleaned we used it into our model to calculate odds ratio based on it and assign the appropriate labels to test data.
Getting rid of the lower frequency words from the training data worked the best for our model's accuracy

Problems faced or assumptions made:
To increase the accuracy of our model we tried using Laplace smoothing to handle the unseen words of test data. We tried various alpha values but we couldn't increase our accuarcy more than our current accuracy. So we dropped the idea of Laplace smoothing.
We also tried to using summation of log of each likelihood probablities to calculate odds ratio but again to our observation it really didn't make much difference to our accuracy.
