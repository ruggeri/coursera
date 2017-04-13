import config
import numpy as np

class FileReader:
    def __init__(self, filename):
        self.filename_ = filename
        self.text_ = None
        self.one_hot_text_ = None

        self.char_to_int_ = None
        self.int_to_char_ = None

        self.vocab_size_ = None

    def text(self):
        if self.text_:
            return self.text_

        if config.TEST_MODE:
            self.text_ = read_test_file()
            return self.text_

        self.text_ = ""
        with open(self.filename_, 'r') as f:
            for line in f.readlines():
                self.text_ += line.lower()
        return self.text_

    # NB: to ensure that chars are mapped to the same ints, we must
    # use the same text. If we permute the text, the mapping will not
    # be the same!
    def set_maps_(self):
        self.int_to_char_ = {}
        self.char_to_int_ = {}
        for char in self.text():
            if char in self.char_to_int_: continue
            char_int = len(self.int_to_char_)
            self.char_to_int_[char] = char_int
            self.int_to_char_[char_int] = char

    def char_to_int(self, char):
        if not self.char_to_int_:
            self.set_maps_()
        return self.char_to_int_[char]

    def int_to_char(self, char_int):
        if not self.int_to_char_:
            self.set_maps_()
        return self.int_to_char_[char_int]

    def vocab_size(self):
        if self.vocab_size_:
            return self.vocab_size_
        self.vocab_size_ = len(set(self.text()))
        return self.vocab_size_

    def one_hot_encode_letter(self, char):
        char_int = self.char_to_int(char)
        one_hot = np.zeros([self.vocab_size()])
        one_hot[char_int] = 1.0
        return one_hot

    def one_hot_text(self):
        if self.one_hot_text_:
            return self.one_hot_text_
        self.one_hot_text_ = np.zeros(
            [len(self.text()), self.vocab_size()]
        )
        for i, char in enumerate(self.text()):
            char_int = self.char_to_int(char)
            self.one_hot_text_[i, char_int] = 1.0

        return self.one_hot_text_

NUM_TEST_CHARS = 1000000
def read_test_file():
    text = ""
    for _ in range(NUM_TEST_CHARS):
        if np.random.random() > 0.9:
            text += "abcdef"
        else:
            text += "ghijkl"
    return text
