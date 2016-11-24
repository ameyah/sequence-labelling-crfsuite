from hw3_corpus_tool import get_utterances_from_filename, get_data
import argparse
import os
import pycrfsuite
from collections import defaultdict

__author__ = 'ameya'


class AdvancedCrf():
    class __AdvancedCrf():
        def __init__(self):
            self.features = []
            self.act_tags = []
            self.raw_data = {}
            self.trainer = pycrfsuite.Trainer(verbose=False)
            self.tagger = pycrfsuite.Tagger()
            self.tag_data = defaultdict(list)

        @staticmethod
        def get_features_act_tags(dialogue):
            features = []
            act_tags = []
            first_utterance = True
            current_speaker = dialogue[0].speaker
            for i, utterance in enumerate(dialogue):
                try:
                    utterance_feature = []
                    if first_utterance:
                        utterance_feature.append("FEATURE_FU")
                        first_utterance = False
                    elif current_speaker != utterance.speaker:
                        utterance_feature.append("FEATURE_SC")
                        current_speaker = utterance.speaker
                    if utterance.pos:
                        utterance_feature.append("LEN_" + str(len(utterance.pos)))

                        utterance_feature.append("TOKEN_FT_" + utterance.pos[0].token)
                        utterance_feature.append("POS_FT_" + utterance.pos[0].pos)
                        previous_token = "TOKEN_" + utterance.pos[0].token
                        previous_pos = "POS_" + utterance.pos[0].pos
                        for word_index, word in enumerate(utterance.pos[1: -1]):
                            if word.pos not in ['.', ',']:
                                current_token = "TOKEN_" + word.token
                                current_pos = "POS_" + word.pos
                                utterance_feature.append(current_token)
                                utterance_feature.append(current_pos)
                            else:
                                continue

                            # append bigrams
                            """
                            try:
                                next_token = "TOKEN_" + utterance.pos[word_index + 2].token
                                next_pos = "POS_" + utterance.pos[word_index + 2].pos
                            except IndexError as e:
                                next_token = ""
                                next_pos = ""
                            """
                            utterance_feature.append("%s_%s" % (previous_token, current_token))
                            utterance_feature.append("%s_%s" % (previous_pos, current_pos))
                            """
                            utterance_feature.append("%s_%s" % (current_token, next_token))
                            utterance_feature.append("%s_%s" % (current_pos, next_pos))
                            utterance_feature.append("%s_%s_%s" % (previous_token, current_token, next_token))
                            print("%s|%s|%s" % (previous_token, current_token, next_token))
                            utterance_feature.append("%s_%s_%s" % (previous_pos, current_pos, next_pos))
                            """
                            previous_token, previous_pos = current_token, current_pos
                        utterance_feature.append("TOKEN_LT_" + utterance.pos[-1].token)
                        utterance_feature.append("POS_LT_" + utterance.pos[-1].pos)
                        utterance_feature.append("%s_%s" % (previous_token, utterance.pos[-1].token))
                        utterance_feature.append("%s_%s" % (previous_pos, utterance.pos[-1].pos))
                    else:
                        utterance_feature.append("TOKEN_NONE")
                        utterance_feature.append("POS_NONE")
                    try:
                        utterance_feature.append("TOKEN_NT_" + dialogue[i + 1].pos[0].token)
                    except:
                        # print(utterance.text)
                        # print(dialogue[i + 1].text)
                        pass
                    features.append(utterance_feature)
                    act_tags.append(utterance.act_tag)
                except:
                    continue
            # add feature in last utterance
            features[-1].append("FEATURE_LU")
            return features, act_tags

        def scan_input_dir(self, input_dir):
            train_data = get_data(input_dir)
            for dialogue in train_data:
                features, act_tags = self.get_features_act_tags(dialogue[1])
                self.trainer.append(features, act_tags)

        def train_model(self):
            # self.trainer.set_params({'c1': 1.0, 'c2': 0.80, 'max_iterations': 43, 'feature.possible_transitions': True})
            self.trainer.set_params({'c1': 2.0, 'c2': 0.3, 'max_iterations': 90, 'feature.possible_transitions': True})
            self.trainer.train('adv_sequence_label_model.crfsuite')

        def tag_dir(self, test_dir):
            self.tagger.open('adv_sequence_label_model.crfsuite')
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
        if AdvancedCrf.__instance is None:
            AdvancedCrf.__instance = AdvancedCrf.__AdvancedCrf()
        self.__dict__['AdvancedCrf__instance'] = AdvancedCrf.__instance

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
    train_instance = AdvancedCrf()
    args = get_command_args()
    train_instance.scan_input_dir(os.path.abspath(args.input_dir))
    train_instance.train_model()
    train_instance.tag_dir(os.path.abspath(args.test_dir))
    train_instance.write_data(os.path.abspath(args.output_file))
