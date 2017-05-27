import warnings
from asl_data import SinglesData

ERROR_LOGGING = False

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    # get the list of test sequences
    test_sequences = list(test_set.get_all_Xlengths().values())
    for test_X, test_Xlength in test_sequences:
        # create a dictionary of (word, prob) for this test case
        probabilities.append({})
        guesses.append("")
        # find the score for each model (each model has a word it is trained for)
        for word, model in models.items():
            try:
                score = model.score(test_X, test_Xlength)
                # store set of probabilities, we'll find the best one later
                probabilities[-1][word] = score
            except Exception as e:
                if ERROR_LOGGING:
                    print(str(e))
                # if a probability cannot be calculated, set to -inf instead
                probabilities[-1][word] = float("-infinity")
                continue;
        # find the best guess
        guesses[-1] = max(probabilities[-1], key=probabilities[-1].get)

    return (probabilities, guesses)

    """
    for word_id in range (0, len(test_set.get_all_Xlengths())):
        X, lengths = test_set.get_word_Xlengths[word_id]
        current_sequence = test_set.get_item_sequences(word_id)
        print (current_sequence)
        current_length = test_set.get_item_Xlengths(word_id)
        for word, model in models.items():
            score = model.score(current_sequence, current_length)
            if (probabilities[word_id] < p):
                probabilities[word_id] = p
                guesses[word_id] = word
    raise NotImplementedError
    """

    """
    for word, model in models.items():
        word_id = test_set.wordlist.index(word)
        print (word)
        print (word_id)
        print(model.score(test_set.get_item_Xlengths(word_id)))
        if (probabilities[word_id] < p):
            probabilities[word_id] = p
            guesses[word_id] = word
    """
    raise NotImplementedError
    return (probabilities, guesses)
