from collections import Counter
from numpy import dot, multiply, array, array_equal, copy
from os import listdir
from os.path import isfile
import matplotlib.pyplot as plt
from sys import maxsize


class Perceptron:
    """
    Machine Learning Perceptron Algorithm that trains a linear model to tell spam emails.

    Given labelled trainning dataset, it keeps adjust the weight of the model
    until all the emails are correctly classified.
    Otherwise, it adds the feature vector of the email that is mistakenly classified to the weight.

    Args
    --------
    train_file,valid_file,test_file: str,str,str
        name of file of training set,validation set and testing set

    Attributes
    --------
    train_set, valid_set, test_set: list,list,list
        list of emails(str) 
    train_label, valid_label, test_label: list,list,list
        list of labels(int): 1 or -1
    word_list: list
        list of vocabularies in trainiing set

    Methods
    --------
    words(data,X):
        Generate vocabularies from 'data' with minial occurance of 'X' times
    feature_vector(email):
        Generate the feature vector of an email based on word_list
    perceptron_train(data,maxiter):
        Return the weight, the mistake times and the iteration times to train the model
    perceptron_error(w,data):
        Return the rate of error times/total dataset of applying the model to the dataset
    """

    def __init__(self, train_file, valid_file, test_file):

        self.train_set = open(train_file, 'r').readlines()
        self.valid_set = open(valid_file, 'r').readlines()
        self.test_set = open(test_file, 'r').readlines()

        # data_sets
        self.train_label = [1 if int(
            email[0]) == 1 else -1 for email in self.train_set]
        # self.train_set = [email[2:].strip() for email in self.train_set]
        self.train_set = list(map(lambda x: x[2:].strip(), self.train_set))

        self.valid_label = [1 if int(
            email[0]) == 1 else -1 for email in self.valid_set]
        self.valid_set = list(map(lambda x: x[2:].strip(), self.valid_set))

        self.test_label = [1 if int(
            email[0]) == 1 else -1 for email in self.test_set]
        self.test_set = list(map(lambda x: x[2:].strip(), self.test_set))

        print("Finish preparing data sets.")

        # Build vocabulary list from the training set
        self.word_list = self.words(self.train_set, 25)

        print("Finish preparing wordlist")

    def words(self, data, X):
        """
        Args
        ------
            data:[list of emails, list of labels]
            X: int
                Minim occurance of a word in separate emails

        Return
        ------
            word list: list of str
        """

        # content = Counter(' '.join(data).split())
        alpha = {}
        for email in data:
            words = Counter(email.strip().split())
            for word in words:
                alpha[word] = alpha.get(word, 0) + 1

        return [key for key, value in alpha.items() if value > X]
        # [key for key, value in content.items() if value >= X]

    def feature_vector(self, email):
        """
        Args
        ------
            data:[list of emails, list of labels]
            X: int
                Minim occurance of a word in separate emails

        Return
        ------
            word list: list of str
        """
        words = email.split()
        return [1 if word in words else 0 for word in self.word_list]

    def perceptron_train(self, data, maxiter=maxsize):
        """
        Perceptrion Algorithm

        Start with the weight as list of zeros. 
        Classify the email with its feature vector
        Whenver the result is inconsistent with the label(dot(w,x)*y<=0),
        adjust the weight by adding the vector.
        Repeat the process util it classifies all the emails correctly


        Args
        ------
            data:[list of emails, list of labels]
            maxiter: int
                Maximum time allowed for iteration

        Return
        ------
            weight: list of int that corresponds to word list
        """
        emails = data[0]
        labels = data[1]

        flag = False
        w = array([0] * len(self.word_list))
        k = 0
        iters = 0

        while (not flag) and (iters < maxiter):
            prev_w = copy(w)
            flag = True

            for i, email in enumerate(emails):

                x = self.feature_vector(email)
                y = labels[i]

                if multiply(dot(w, x), y) > 0:  # Succint Equation
                    continue
                # if (dot(w, x) > 0 and y == 1) or (dot(w, x) <= 0 and y == -1): #Original Equation
                #     continue
                else:
                    w += multiply(x, y)
                    # flag = False
                    print(w)

                # Output the weight list if it is not changed through an iteration
                # As there are emails whose feacture vector is all zeros which invalidates the judgement of w*x*y>0,
                # the measurement of the model changes its focuses on weights
                # rather than classification equation.
                if not array_equal(prev_w, w):
                    prev_w = copy(w)
                    flag = False
                    k += 1
                    print(k)

            iters += 1
            print("time %d" % iters)

        return w, k, iters

    def perceptron_error(self, w, data):
        """
        Args
        ------
            w: list of int
                weight
            data:[list of emails, list of labels]
                validation of testing dataset
        Return
        ------
            error rate: a float = mistakes/total
        """
        emails = data[0]
        labels = data[1]
        k = 0
        for i, email in enumerate(emails):
            x = self.feature_vector(email)
            y = labels[i]
            if dot(w, x) * y <= 0:
                k += 1
                print(k)
        return k / len(emails)

if __name__ == '__main__':

    folder = 'percept_data/'
    filename_train = folder + 'spam_train.txt'
    filename_test = folder + 'spam_test.txt'

    # Prepare validation dataset file
    if not isfile("valid.txt"):
        with open(filename_train) as f:
            with open("valid.txt", 'w') as validf:
                validf.write(''.join(f.readlines()[-1000:]))
                validf.close()
            f.close()

    ml = Perceptron("train.txt", "valid.txt", filename_test)

    if not isfile(folder + "word.txt"):
        with open(folder + "word.txt", 'w') as file:
            file.write('\n'.join(ml.word_list))
            file.close()

    if not isfile("weights_4000.txt"):
        w, k, iters = ml.perceptron_train([ml.train_set, ml.train_label])
        print(w)
        print(k)
        print(iters)
        with open("weights_4000.txt", 'w') as file:
            file.write('\n'.join(list(str(k) for k in w)))
            file.close()

    with open("weights_4000.txt", 'r') as file:
        weights = list(map(lambda x: int(x), file.readlines()))
        file.close()

    # if not isfile("output.txt"):
    #     file = open("output.txt","w")
    #     file.write("{},{},{}".format("Data Amount","Word Filter","Iteration Maximum"))
    #     file.close()

    #     with open("output.txt",'a') as file:
    #         file.write("{},{},{}".format(str(len(ml.train_set))),)
    #         file.write('\n')
    #         file.write(str(k))
    #         file.write('\n')
    #         file.write(str(iters))
    #         file.close()

    # Q3
    # 453 mistakes were made through 11 iterations.
    # validation error is 0.019
    # valid_error = ml.perceptron_error(weights, [ml.valid_set,ml.valid_label])
    # print(valid_error)

    # Q4
    """
    The most positive words are ['sight', 'remov', 'market', 'click', 'present', 'our', 'pleas', 'deathtospamdeathtospamdeathtospam'] with weight of [19, 18, 16, 15, 15, 14, 14, 13]
    The most positive words are ['wrote', 'prefer', 'll', 'i', 'reserv', 're', 'set', 'someth'] with weight of [-16, -16, -13, -12, -12, -11, -11, -11]

    """
    # pos_weights = [weights[i] for i in array(weights).argsort()[-8:][::-1]]
    # pos_words = [ml.word_list[i] for i in array(weights).argsort()[-8:][::-1]]

    # neg_weights = [weights[i] for i in array(weights).argsort()[:8]]
    # neg_words = [ml.word_list[i] for i in array(weights).argsort()[:8]]

    # print("The most positive words are {} with weight of {}".format(
    #     pos_words, pos_weights))
    # print("The most negative words are {} with weight of {}".format(
    #     neg_words, neg_weights))

    # Q5
    # Test error for different Data amount
    # iteration times =  = [7,5,5,9]
    # error rate = [0.089,0.04,0.029,0,021]

    # n = [100, 500, 1000, 2000, 4000]
    # for n_i in n:
    #     partial_data = [ml.train_set[:n_i], ml.train_label[:n_i]]
    #     w, k, iters = ml.perceptron_train(partial_data)
    #     e = ml.perceptron_error(w, [ml.valid_set, ml.valid_label])
    #     with open("dataamount" + str(n_i) + ".txt", 'w') as file:
    #         file.write('\n'.join(list(str(k) for k in w)))
    #         print(str(iters))
    #         print(str(e))
    #         print('end')
    #         file.close()

    # plt.close('all')
    # f,axarr = plt.subplots(2,sharex=True)
    # axarr[0].scatter(n, errors)
    # axarr[0].set_title('error-amount times')
    # for i,txt in enumerate(errors):
    #     axarr[0].annotate(txt,(n[i],errors[i]))
    # axarr[1].scatter(n,iter_list)
    # for i,txt in enumerate(iter_list):
    #     axarr[1].annotate(txt,(n[i],iter_list[i]))
    # axarr[1].set_title('iter-amount times')
    # plt.show()

    # ws = []
    iters_list = []
    errors = []
    Xs = [20, 30]
    for iter_max in range(5, 9):
        for x in Xs:
            ml.word_list = ml.words(ml.train_set, x)
            w, k, iters = ml.perceptron_train(
                [ml.train_set, ml.train_label], maxiter=iter_max)
            valid_error = ml.perceptron_error(
                w, [ml.valid_set, ml.valid_label])
        #     ws.append(w)
            iters_list.append(min(iter_max, iters))
            errors.append(valid_error)
            Xs.append(x)
            print("iteration={}, X={}, validation error = {}".format(
                min(iter_max, iters), x, valid_error))

    #     ml.word_list = ml.words(ml.train_set, 30)
    #     w, k, iters = ml.perceptron_train(
    #         [ml.train_set, ml.train_label], maxiter=iter_max)
    #     valid_error = ml.perceptron_error(
    #         w, [ml.valid_set, ml.valid_label])
    #     ws.append(w)
    #     iters_list.append(iter_max)
    #     es.append(valid_error)
    #     Xs.append(30)
    #     print("iteration={}, X=30, validation error = {}".format(
    #         iter_max, valid_error))

        with open('configurations.txt', 'a') as file:
            # for i in array(es).argsort()[:4]:
            # for i in es:
            file.write(errors[i])
            # file.write(' '.join(ws[i]))
            file.write(iters_list[i])
            file.write(Xs[i])
            file.close()
