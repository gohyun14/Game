import scipy.spatial.distance

from players.guesser import Guesser


class AIGuesser(Guesser):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.num = 0

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue
        self.num = num
        print("The clue is:", clue, num)
        li = [clue, num]
        return li

    def keep_guessing(self):
        return self.num > 0

    def get_answer(self):
        cache =  len(self.word_vectors) < 400
        sorted_words = self.compute_distance(self.clue, self.words, cache)
        print(f'guesses: {sorted_words}')
        self.num -= 1
        return sorted_words[0][1]

    def compute_distance(self, clue, board, cache):
        w2v = []

        for word in board:
            try:
                if word[0] == '*':
                    continue
                if cache:
                    w2v.append((self.word_vectors['dict_word_dists'][word.lower()][clue], word))
                else:
                    w2v.append((scipy.spatial.distance.cosine(self.word_vectors['dict_word_dists'][clue],
                                                            self.word_vectors['dict_word_dists'][word.lower()]), word))
            except KeyError:
                continue

        w2v = list(sorted(w2v))
        return w2v

