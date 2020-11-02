import numpy as np
from collections import defaultdict
import pandas as pd


class NaiveBayes:

    def __init__(self, unique_classes):
        self.classes = unique_classes  # Constructor is simply passed with unique number of classes of the training set
        self.bow_dicts = np.array([defaultdict(lambda: 0) for _ in range(self.classes.shape[0])])

    def addToBow(self, example, dict_index):
        """
            Parameters:
            1. example
            2. dict_index - implies to which BoW category this example belongs to
            What the function does?
            -----------------------
            It simply splits the example on the basis of space as a tokenizer and adds every tokenized word to
            its corresponding dictionary/BoW
            Returns:
            ---------
            Nothing
       """

        for token_word in example.split():  # for every word in preprocessed example
            self.bow_dicts[dict_index][token_word] += 1  # increment in its count

    def train(self, x_train, y_train):
        """
            Parameters:
            1. dataset - shape = (m X d)
            2. labels - shape = (m,)
            What the function does?
            -----------------------
            This is the training function which will train the Naive Bayes Model i.e compute a BoW for each
            category/class.
            Returns:
            ---------
            Nothing
        """

        # constructing BoW for each category
        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = x_train[y_train == cat]  # filter all examples of category == cat
            cleaned_examples = pd.DataFrame(all_cat_examples)

            # now construct BoW of this particular category
            np.apply_along_axis(self.addToBow, 1, cleaned_examples, cat_index)

        ###################################################################################################

        '''
            Although we are done with the training of Naive Bayes Model BUT!!!!!!
            ------------------------------------------------------------------------------------
            Remember The Test Time Formula ? : {for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ] } * p(c)
            ------------------------------------------------------------------------------------

            We are done with constructing of BoW for each category. But we need to precompute a few 
            other calculations at training time too:
            1. prior probability of each class - p(c)
            2. vocabulary |V| 
            3. denominator value of each class - [ count(c) + |V| + 1 ] 

            Reason for doing this precomputing calculations stuff ???
            ---------------------
            We can do all these 3 calculations at test time too BUT doing so means to re-compute these 
            again and again every time the test function will be called - this would significantly
            increase the computation time especially when we have a lot of test examples to classify!!!).  
            And moreover, it does not make sense to repeatedly compute the same thing - 
            why do extra computations ???
            So we will precompute all of them & use them during test time to speed up predictions.

        '''

        ###################################################################################################
        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            # Calculating prior probability p(c) for each class
            prob_classes[cat_index] = np.sum(y_train == cat) / float(y_train.shape[0])

            # Calculating total counts of all the words of each class
            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(
                np.array(count)) + 1  # |v| is remaining to be added

            # get all words of this category
            all_words += self.bow_dicts[cat_index].keys()

        # combine all words of every category & make them unique to get vocabulary -V- of entire training set

        vocab = np.unique(np.array(all_words))
        self.vocab_length = vocab.shape[0]

        # computing denominator value
        denom = np.array(
            [cat_word_counts[cat_index] + self.vocab_length + 1 for cat_index, cat in enumerate(self.classes)])

        '''
            Now that we have everything precomputed as well, its better to organize everything in a tuple 
            rather than to have a separate list for every thing.

            Every element of self.cats_info has a tuple of values
            Each tuple has a dict at index 0, prior probability at index 1, denominator value at index 2
        '''

        self.cats_info = np.array([(self.bow_dicts[cat_index],
                                    prob_classes[cat_index],
                                    denom[cat_index]) for cat_index, cat in enumerate(self.classes)])

    def getExampleProb(self, test_example):

        """
            Parameters:
            -----------
            1. a single test example
            What the function does?
            -----------------------
            Function that estimates posterior probability of the given test example
            Returns:
            ---------
            probability of test example in ALL CLASSES
        """

        likelihood_prob = np.zeros(self.classes.shape[0])  # to store probability w.r.t each class

        # finding probability w.r.t each class of the given test example
        for cat_index, cat in enumerate(self.classes):

            for test_token in test_example.split():  # split the test example and get p of each test word

                ####################################################################################

                # This loop computes : for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]

                ####################################################################################

                # get total count of this test token from it's respective training dict to get numerator value
                test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1

                # now get likelihood of this test_token word
                test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])

                # remember why taking log? To prevent underflow!
                likelihood_prob[cat_index] += np.log(test_token_prob)

        # we have likelihood estimate of the given example against every class but we need posterior probability
        post_prob = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

        return post_prob

    def test(self, test_set):

        """
            Parameters:
            -----------
            1. A complete test set of shape (m,)

            What the function does?
            -----------------------
            Determines probability of each test example against all classes and predicts the label
            against which the class probability is maximum
            Returns:
            ---------
            Predictions of test examples - A single prediction against every test example
        """

        predictions = []  # to store prediction of each test example
        for example in test_set:

            # simply get the posterior probability of every example
            post_prob = self.getExampleProb(example)  # get prob of this example for both classes

            # simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])

        return np.array(predictions)
