from hw3_corpus_tool import get_utterances_from_filename
import argparse
from collections import defaultdict
import os
from math import ceil
from random import shuffle

__author__ = 'ameya'


class BaselineCrf():
    class __BaselineCrf():
        def __init__(self):
            self.training_dir = None
            self.spam_files = 0
            self.ham_files = 0
            self.train_less = 0
            self.less_files = 0
            self.less_spam_files = 0
            self.less_ham_files = 0
            self.train_iterations = 0
            self.files_dict = {}
            self.weights = {}
            self.cache_feature_dict = {}
            self.bias = 0
            self.spam_label = -1
            self.ham_label = 1

        def set_training_dir(self, training_dir):
            self.training_dir = training_dir

        def set_train_type(self, train_less, train_iterations, **kwargs):
            self.train_less = train_less
            self.train_iterations = train_iterations
            if kwargs['spam_label']:
                self.spam_label = kwargs['spam_label']
            if kwargs['ham_label']:
                self.ham_label = kwargs['ham_label']
            if self.train_less != 0:
                self.compute_files()

        def compute_files(self):
            for current_dir, dirnames, filenames in os.walk(self.training_dir):
                last_dir_name = os.path.basename(current_dir)
                if last_dir_name == "spam":
                    for file_name in filenames:
                        file_extension = os.path.splitext(file_name)[1]
                        if file_extension == ".txt":
                            self.spam_files += 1
                elif last_dir_name == "ham":
                    for file_name in filenames:
                        file_extension = os.path.splitext(file_name)[1]
                        if file_extension == ".txt":
                            self.ham_files += 1
            if self.train_less != 0:
                self.less_spam_files = ceil((self.train_less / 100) * self.spam_files)
                self.less_ham_files = ceil((self.train_less / 100) * self.ham_files)
                self.spam_files = 0
                self.ham_files = 0

        # procedure to recursively determine all the spam and ham directories
        # and store in spam_dirs and ham_dirs
        def map_spam_ham_dirs(self):
            for current_dir, dirnames, filenames in os.walk(self.training_dir):
                if self.train_less != 0:
                    if "train/2" not in current_dir:
                        continue
                last_dir_name = os.path.basename(current_dir)
                if last_dir_name == "spam":
                    for file_name in filenames:
                        file_extension = os.path.splitext(file_name)[1]
                        if file_extension == ".txt":
                            if self.train_less != 0:
                                if self.less_spam_files == 0:
                                    continue
                                self.less_spam_files -= 1
                            full_file_path = os.path.join(current_dir, file_name)
                            self.files_dict[full_file_path] = self.spam_label
                            # self.cache_features(os.path.join(current_dir, file_name))
                            feature_dict = defaultdict(int)
                            with open(full_file_path, "r", encoding="latin1") as file_handler:
                                file_content = file_handler.read()
                                features = file_content.split()
                                for feature in features:
                                    feature_dict[feature] += 1
                                    self.weights[feature] = 0
                            self.cache_feature_dict[full_file_path] = feature_dict
                            self.spam_files += 1
                elif last_dir_name == "ham":
                    for file_name in filenames:
                        file_extension = os.path.splitext(file_name)[1]
                        if file_extension == ".txt":
                            if self.train_less != 0:
                                if self.less_ham_files == 0:
                                    continue
                                self.less_ham_files -= 1
                            full_file_path = os.path.join(current_dir, file_name)
                            self.files_dict[full_file_path] = self.ham_label
                            # self.cache_features(os.path.join(current_dir, file_name))
                            feature_dict = defaultdict(int)
                            with open(full_file_path, "r", encoding="latin1") as file_handler:
                                file_content = file_handler.read()
                                features = file_content.split()
                                for feature in features:
                                    feature_dict[feature] += 1
                                    self.weights[feature] = 0
                            self.cache_feature_dict[full_file_path] = feature_dict
                            self.ham_files += 1

        def train_model(self):
            files_dict_keys = list(self.files_dict.keys())
            for i in range(0, self.train_iterations):
                shuffle(files_dict_keys)
                self.perceptron_train(files_dict_keys)

        def perceptron_train(self, files_dict_keys):
            for file_key in files_dict_keys:
                feature_dict = self.cache_feature_dict[file_key]
                activation = 0
                for feature in feature_dict:
                    activation += (self.weights[feature] * feature_dict[feature])
                activation += self.bias
                file_key_label = self.files_dict[file_key]
                if file_key_label * activation <= 0:
                    # wrong prediction. Adjust the weightss
                    for feature in feature_dict:
                        self.weights[feature] += (file_key_label * feature_dict[feature])
                    self.bias += file_key_label

        def write_training_data(self, write_file):
            try:
                with open(write_file, "w", encoding='latin1') as file_handler:
                    # the first line is for "bias"
                    file_handler.write(str(self.bias) + '\n')
                    # following lines contain weights of features
                    for weight in self.weights:
                        try:
                            file_handler.write(str(weight) + ' ' + str(self.weights[weight]) + '\n')
                        except:
                            # print("Exception in writing weights")
                            continue
            except:
                return

    __instance = None

    def __init__(self):
        if BaselineCrf.__instance is None:
            BaselineCrf.__instance = BaselineCrf.__BaselineCrf()
        self.__dict__['BaselineCrf__instance'] = BaselineCrf.__instance

    def __getattr__(self, attr):
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__instance, attr, value)


def get_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help='Directory of input training data.')
    parser.add_argument("-l", "--less", default=0, type=int, help='Add if data is to be trained with only x% data')
    parser.add_argument("-i", "--iterations", default=20, type=int, help='Add if # of iterations need to be changed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train_instance = BaselineCrf()
    args = get_command_args()
    train_instance.set_training_dir(os.path.abspath(args.input_dir))
    train_instance.set_train_type(args.less, args.iterations, spam_label=-1, ham_label=1)
    train_instance.map_spam_ham_dirs()
    train_instance.train_model()
    train_instance.write_training_data('per_model.txt')

get_utterances_from_filename('labeled data/0001.csv')
