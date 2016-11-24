import glob
from hw3_corpus_tool import get_utterances_from_filename, get_data
import argparse
import os
import pycrfsuite
from collections import defaultdict

__author__ = 'ameya'


class BaselineCrf():
    class __BaselineCrf():
        def __init__(self):
            self.features = []
            self.act_tags = []
            self.raw_data = {}
            self.trainer = pycrfsuite.Trainer(verbose=False)
            self.tagger = pycrfsuite.Tagger()
            self.tag_data = defaultdict(list)

        def read_all_files(self, input_dir):
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".csv"):
                    utterances_list = get_utterances_from_filename(os.path.join(input_dir, file_name))
                    # self.raw_data[]

        @staticmethod
        def get_features_act_tags(dialogue):
            features = []
            act_tags = []
            first_utterance = True
            current_speaker = dialogue[0].speaker
            for utterance in dialogue:
                utterance_feature = []
                if first_utterance:
                    utterance_feature.append("FEATURE_FU")
                    first_utterance = False
                elif current_speaker != utterance.speaker:
                    utterance_feature.append("FEATURE_SC")
                    current_speaker = utterance.speaker
                if utterance.pos:
                    utterance_feature.extend(["TOKEN_" + word.token for word in utterance.pos])
                    utterance_feature.extend(["POS_" + word.pos for word in utterance.pos])
                if utterance.act_tag:
                    features.append(utterance_feature)
                    act_tags.append(utterance.act_tag)
            return features, act_tags

        def scan_input_dir(self, input_dir):
            train_data = get_data(input_dir)
            for dialogue in train_data:
                features, act_tags = self.get_features_act_tags(dialogue[1])
                self.trainer.append(features, act_tags)

        def train_model(self):
            # self.trainer.set_params({'c1': 1.0, 'c2': 0.80, 'max_iterations': 43, 'feature.possible_transitions': True})
            self.trainer.set_params({'c1': 2.0, 'c2': 0.1, 'max_iterations': 130, 'feature.possible_transitions': True})
            self.trainer.train('sequence_label_model.crfsuite')

        def tag_dir(self, test_dir):
            self.tagger.open('sequence_label_model.crfsuite')
            test_data = get_data(test_dir)
            for dialogue in test_data:
                utterances = dialogue[1]
                features, act_tags = self.get_features_act_tags(utterances)
                self.tag_data[dialogue[0]].extend(self.tagger.tag(features))

        def write_data(self, write_file):
            # print(self.tag_data)
            try:
                with open(write_file, "w", encoding='latin1') as file_handler:
                    for file_name in self.tag_data:
                        try:
                            file_handler.write("Filename=\"" + str(os.path.basename(file_name)) + "\"" + '\n')
                            for tag in self.tag_data[file_name]:
                                file_handler.write(str(tag) + '\n')
                            file_handler.write('\n')
                        except:
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
    parser.add_argument("test_dir", help='Directory of test data.')
    parser.add_argument("output_file", help='Output file name.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train_instance = BaselineCrf()
    args = get_command_args()
    train_instance.scan_input_dir(os.path.abspath(args.input_dir))
    train_instance.train_model()
    train_instance.tag_dir(os.path.abspath(args.test_dir))
    train_instance.write_data(os.path.abspath(args.output_file))
