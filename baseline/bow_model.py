import numpy as np
import pickle
from os.path import join
from os import listdir

import util
from util import encode, get_all_samples_from_adir


class BowModel(object):

    def __init__(self, traindir="../data/train"):
        self.traindir = traindir
        self.author_means = self.get_author_mean_encodings()

    def get_mean_encoding(self, authorname):
        samples = get_all_samples_from_adir(authorname)
        encodings = [encode(sample) for sample in samples]
        mean = np.mean(np.array(encodings), axis=0)
        return mean


    def get_author_mean_encodings(self):

        def stripped(text):
            new = []
            for c in text:
                if c != '_':
                    new.append(c)
            return str(new)

        authornames = [an for an in listdir(self.traindir) if an[0] != '.']

        author_means = {}
        for an in authornames:
            mean = self.get_mean_encoding(an)
            normalized = mean / np.linalg.norm(mean)
            author_means[an] = normalized

        return author_means

    def predict(self, x):
        """ Numpy 1D-12 vector x. Find the author whose mean it is closest to.
        """
        x /= np.linalg.norm(x)
        scores = [(an, x.dot(self.author_means[an])) for an in self.author_means]
        sorted_scores = sorted(scores, reverse=True, key=lambda x: x[1])

        best_author, best_score = sorted_scores[0]
        return best_author

