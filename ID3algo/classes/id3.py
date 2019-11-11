import pandas
import math
import operator


class Id3:
    def __init__(self):
        self.root = 'fast'
        self.data_headers = ['engine', 'turbo', 'weight', 'fuel_eco', 'fast']
        self.data = pandas.read_csv("classes/id3_data.csv", names=self.data_headers)

    @staticmethod
    def reduce_training_data_set(data, delete_rows, selected_root_node):
        for item in delete_rows:
            data = data[data.eval(selected_root_node) != item]
        return data

    def root_node_election(self, target_entropy):
        info_gain_dict = dict()
        attr_entropy, attr_row_del_dict = 0, []
        for x in self.data_headers[:-1]:
            attr_entropy, attr_row_del_dict = self.entropy(x)
            info_gain = self.information_gain(target_entropy, attr_entropy)
            print("IG of ", x, " is ", info_gain)
            info_gain_dict[x] = info_gain
        maxi = max(info_gain_dict.items(), key=operator.itemgetter(1))[0] if len(info_gain_dict.items()) else None
        return maxi, attr_row_del_dict

    def entropy(self, attr_name):
        entropy = 0
        row_del_dict = dict()
        inner_row_del_dict = dict()
        attr_values = list(self.data[attr_name])
        if attr_name == self.root:
            all_classes = list(set(attr_values))
            for x in all_classes:
                probability = attr_values.count(x) / len(attr_values)
                entropy -= (probability * math.log2(probability))
        else:
            merged_class = list(zip(attr_values, list(self.data[self.root])))
            all_classes = list(set(merged_class))
            for x in all_classes:
                probability = merged_class.count(x) / attr_values.count(x[0])
                sub_entropy = probability * math.log2(probability)
                inner_row_del_dict[x[0]] = sub_entropy, x[1]
                row_del_dict[attr_name] = inner_row_del_dict
                entropy -= (attr_values.count(x[0]) / len(attr_values)) * sub_entropy
        return entropy, row_del_dict

    @staticmethod
    def information_gain(target_entropy, attr_entropy):
        info_gain = target_entropy - attr_entropy
        return info_gain
