from hw3_corpus_tool import get_utterances_from_filename
import argparse
import os
from collections import defaultdict

__author__ = 'ameya'

class CrfModelEvaluate():
    class __CrfModelEvaluate():
        def __init__(self):
            self.tagged_data = defaultdict(list)
            self.correctly_classified_tags = 0
            self.total_tags = 0

        def store_labels(self, output_file):
            current_file = None
            with open(output_file, "r", encoding="latin1") as file_handler:
                for line in file_handler:
                    if "Filename=\"" in line:
                        current_file = line.split("Filename=\"")[1]
                        current_file = current_file[:-2]
                    elif line.strip() != '':
                        if current_file:
                            self.tagged_data[current_file].append(line.strip())

        def evaluate(self, test_dir):
            for file_name in os.listdir(test_dir):
                if file_name.endswith(".csv"):
                    utterances_list = get_utterances_from_filename(os.path.join(test_dir, file_name))
                    try:
                        tagged_labels = self.tagged_data[file_name]
                    except KeyError as e:
                        print(file_name)
                    for i in range(len(utterances_list)):
                        # print(file_name)
                        if utterances_list[i].act_tag == tagged_labels[i]:
                            self.correctly_classified_tags += 1
                        self.total_tags += 1

            print(self.correctly_classified_tags)
            print(self.total_tags)
            print("Accuracy: " + str(self.correctly_classified_tags/self.total_tags))

    __instance = None

    def __init__(self):
        if CrfModelEvaluate.__instance is None:
            CrfModelEvaluate.__instance = CrfModelEvaluate.__CrfModelEvaluate()
        self.__dict__['CrfModelEvaluate__instance'] = CrfModelEvaluate.__instance

    def __getattr__(self, attr):
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        return setattr(self.__instance, attr, value)


def get_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help='Directory of test data.')
    parser.add_argument("output_file", help='Output file name.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    evaluate_obj = CrfModelEvaluate()
    args = get_command_args()
    evaluate_obj.store_labels(args.output_file)
    evaluate_obj.evaluate(os.path.abspath(args.test_dir))