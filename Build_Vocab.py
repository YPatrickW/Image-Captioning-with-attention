import pickle
import nltk
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_dir, threshold):
    Coco_captions = COCO(json_dir)
    counter = Counter()
    ids = Coco_captions.anns.keys()
    for i, caption_id in enumerate(ids):
        captions = Coco_captions.anns[caption_id]["caption"]
        token = nltk.tokenize.word_tokenize(captions.lower())
        counter.update(token)
        if (i + 1) % 100000 == 0:
            print("[{}/{}] has been tokenized".format(i + 1, len(ids)))
    word_list = []
    for word, freq in counter.items():
        if freq > threshold:
            word_list.append(word)

    Vocabulary_dict = Vocabulary()  # create a vocabulary_dict to enquiry
    Vocabulary_dict.add_word("<pad>")
    Vocabulary_dict.add_word("<start>")
    Vocabulary_dict.add_word("<end>")
    Vocabulary_dict.add_word("<unk>")
    for i, word in enumerate(word_list):
        Vocabulary_dict.add_word(word)

    return Vocabulary_dict


if __name__ == '__main__':
    Vocabulary_dict_train = build_vocab(json_dir="./annotations/captions_train2014.json", threshold=3)
    print("The total length of train Vocabulary_dict is", len(Vocabulary_dict_train))
    store_path = "./Vocabulary_dict/Vocab_train.pkl"
    with open(store_path, "wb") as f:
        pickle.dump(Vocabulary_dict_train, f)
    print("Vocabulary_dict for train has established")
