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

        def scan_input_dir(self, input_dir):
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".csv"):
                    single_file_features = []
                    single_file_act_tags = []
                    utterances_list = get_utterances_from_filename(os.path.join(input_dir, file_name))
                    current_speaker = utterances_list[0].speaker
                    first_utterance = True
                    for utterance in utterances_list:
                        token_list = []
                        pos_tag_list = []
                        utterance_features = []
                        try:
                            for word_tag in utterance.pos:
                                token_list.append("TOKEN_" + str(word_tag.token))
                                if word_tag.pos:
                                    pos_tag_list.append("POS_" + str(word_tag.pos))
                        except TypeError as e:
                            pass
                        if first_utterance:
                            # utterance_features.extend(['1', '0'])
                            utterance_features.extend(['FEATURE_FU'])
                            first_utterance = False
                        elif current_speaker != utterance.speaker:
                            # utterance_features.extend(['0', '1'])
                            utterance_features.extend(['FEATURE_SC'])
                            current_speaker = utterance.speaker
                        utterance_features.extend(token_list)
                        utterance_features.extend(pos_tag_list)
                        single_file_features.append(utterance_features)
                        single_file_act_tags.append(utterance.act_tag)
                    # self.trainer.append(single_file_features, single_file_act_tags)
                    self.features.extend(single_file_features)
                    self.act_tags.extend(single_file_act_tags)
            self.trainer.append(self.features, self.act_tags)

        def train_model(self):
            # self.trainer.set_params({'c1': 1.0, 'c2': 0.80, 'max_iterations': 43, 'feature.possible_transitions': True})
            self.trainer.set_params({'c1': 1.0, 'c2': 0.1, 'max_iterations': 100, 'feature.possible_transitions': True})
            self.trainer.train('sequence_label_model.crfsuite')

        def tag_dir(self, test_dir):
            self.tagger.open('sequence_label_model.crfsuite')
            for file_name in os.listdir(test_dir):
                if file_name.endswith(".csv"):
                    utterances_list = get_utterances_from_filename(os.path.join(test_dir, file_name))
                    current_speaker = utterances_list[0].speaker
                    first_utterance = True
                    for utterance in utterances_list:
                        token_list = []
                        pos_tag_list = []
                        utterance_features = []
                        try:
                            for word_tag in utterance.pos:
                                token_list.append("TOKEN_" + str(word_tag.token))
                                if word_tag.pos:
                                    pos_tag_list.append("POS_" + str(word_tag.pos))
                        except TypeError as e:
                            pass
                        if first_utterance:
                            # utterance_features.extend(['1', '0'])
                            utterance_features.extend(['FEATURE_FU'])
                            first_utterance = False
                        elif current_speaker != utterance.speaker:
                            # utterance_features.extend(['0', '1'])
                            utterance_features.extend(['FEATURE_SC'])
                            current_speaker = utterance.speaker
                        utterance_features.extend(token_list)
                        utterance_features.extend(pos_tag_list)
                        utterance_tag = self.tagger.tag([utterance_features])
                        self.tag_data[file_name].append(utterance_tag)

        def write_data(self, write_file):
            # print(self.tag_data)
            try:
                with open(write_file, "w", encoding='latin1') as file_handler:
                    for file_name in self.tag_data:
                        try:
                            file_handler.write("Filename=\"" + str(file_name) + "\"" + '\n')
                            for tag in self.tag_data[file_name]:
                                file_handler.write(str(tag[0]) + '\n')
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
