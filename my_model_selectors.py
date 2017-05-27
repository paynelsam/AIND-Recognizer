import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

ERROR_LOGGING = False

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        best_num = 0
        min_bic = float("infinity")
        # find the number of hidden states that seems best
        # build a model for each possible number of hidden states
        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                #train a model with i states using self.X and self.lengths
                model = self.base_model(i)
                logL = model.score(self.X, self.lengths) # score the model using the data

                ## find free params in this model ##
                # transition probs = the size of the transmat matrix less
                # one row because they add up to 1 and therefore the final
                # row is deterministic,
                free_transition_probs = i * (i - 1)
                # free means = number of components * features
                free_means = i * len(self.X[0])
                # free covars = size of the covars matrix (components x features)
                free_covars = i * len(self.X[0])
                # starting probs = the size of startprob minus 1 because it adds
                # to 1.0
                free_starting_probs = i - 1
                # sum all components together
                num_free_params = free_transition_probs + free_means + free_covars + free_starting_probs

                # calculate the BIC score
                BIC = -2 * logL + num_free_params * math.log(len(self.X))

                # if this has the best bic score, save it
                if (BIC < min_bic):
                    best_num = i;
                min_bic = min(min_bic, BIC)

            except Exception as e:
                if ERROR_LOGGING:
                    print(str(e))
                continue;

        return self.base_model(best_num)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores

        max_dic = float("-infinity")
        best_num = 0;

        # model to train
        hmm_model = None

        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                # average score of other words
                other_LogL = 0.0
                # number of words - used for averaging later
                word_count = 0

                # train initial HMM on all samples
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag",
                                n_iter=1000, random_state=self.random_state,
                                verbose=False).fit(self.X, self.lengths)
                # score model on all samples
                LogL = hmm_model.score(self.X, self.lengths)

                # score for all words except the current word
                for word in self.hwords:
                    if word == self.this_word:
                        continue
                    X, lengths = self.hwords[word]
                    other_LogL += hmm_model.score(X, lengths)
                    word_count += 1

                # average score of other words
                other_LogL /= float(word_count)

                # compute DIC
                dic = LogL - other_LogL
                if (max_dic < dic):
                    best_num = num_states
                    max_dic = max(max_dic, dic)

            except Exception as e:
                if ERROR_LOGGING:
                    print(str(e))
                continue;
        return self.base_model(best_num)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)



        if(len(self.lengths) >= 2):
            split_method = KFold(n_splits=min(3,len(self.lengths)))
        else :
            split_method = None

        # TODO implement model selection using CV
        best_num = 0
        max_logL = float("-infinity")
        # find the number of hidden states that seems best
        # build a model for each possible number of hidden states
        for i in range(self.min_n_components, self.max_n_components+1):
            # Evaluate each model by training on a set of training sets,
            # then evaluating based on a test set.
            # The model with the best average score over all sets wins.

            j = 0 # number of training/test slices - we'll count as we go
            logL = 0 #initialize the total logL to zero (we'll average this later)

            # split sequences into training and test sets
            if (len(self.lengths) >= 2) :
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    try:
                        # build sequences from indeces collected from split method
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                        # train model on training sequences
                        model = GaussianHMM(n_components=i, n_iter=1000).fit(X_train, lengths_train)

                        # score model on test sequences
                        logL += model.score(X_test, lengths_test)
                        j += 1
                    except Exception as e:
                        # we may hit this for a number of reasons:
                        # for example, there are more parameters than samples
                        if ERROR_LOGGING:
                            print(str(e))
                        continue;
            else:
                try:
                    model = self.base_model(i)
                    logL += model.score(self.X, self.lengths)
                    j += 1
                except Exception as e:
                    if ERROR_LOGGING:
                        print(str(e))
                    continue;

            if (j == 0):
                continue;
            # find average score
            avg_logL = logL/j

            # if this is the best score so far, save it - higher is better
            if(avg_logL > max_logL):
                best_num = i
                max_logL = avg_logL

        # build best model from best number of states, and return it
        best_model = self.base_model(best_num) # this is fitted with self.X and self.lenghts
        return best_model


