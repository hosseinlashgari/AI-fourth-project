from collections import defaultdict
import dill


def preprocess(datalist: list):
    useless = ["*", "(", ")", "=", "&", "%", "$", "#", ",", ".", "'", ":", '"', "{", "}", ";", "/", "|", ">", "<", "?"]
    preprocessed = []
    for word in datalist:
        flag = True
        for less in useless:
            if word.__contains__(less):
                flag = False
                break
        if flag:
            preprocessed.append(word)
    return preprocessed


def make_dict(dataset, mode: int = 0):
    dataset_list = []
    for line in dataset:
        dataset_list.extend(list(line.split()))
    dataset_list = preprocess(dataset_list)
    uni_dict = defaultdict(lambda: 0)
    bi_dict = defaultdict(lambda: 0)
    for word in dataset_list:
        uni_dict[word] += 1
    if mode:
        repetitive = []
        for m in range(10):
            max_key = max(uni_dict, key=lambda k: uni_dict[k])
            repetitive.append(max_key)
            del uni_dict[max_key]
        for line in dataset:
            line_list = preprocess(list(line.split()))
            for word in range(len(line_list) - 1):
                if repetitive.count(line_list[word]) == 0 and repetitive.count(line_list[word + 1]) == 0:
                    bi_dict[line_list[word] + line_list[word + 1]] += 1
        return uni_dict, bi_dict, repetitive
    else:
        for line in dataset:
            line_list = preprocess(list(line.split()))
            for word in range(len(line_list) - 1):
                bi_dict[line_list[word] + line_list[word + 1]] += 1
        return uni_dict, bi_dict, None


def bigram(uni_dict: defaultdict, bi_dict: defaultdict, word1: str, word2: str):
    m = sum(uni_dict.values())
    max_value = max(uni_dict.values())
    uni_prob = uni_dict[word2] / m
    bi_prob = 0
    if uni_dict[word1]:
        bi_prob = bi_dict[word1 + word2] / uni_dict[word1]
    lambda1 = uni_dict[word1] / max_value / 100
    lambda3 = (1 - uni_dict[word2] / max_value) / 1000
    lambda2 = 1 - lambda1 - lambda3
    epsilon = 1000 / m
    return lambda1 * bi_prob + lambda2 * uni_prob + lambda3 * epsilon


def unigram(uni_dict: defaultdict, word: str):
    m = sum(uni_dict.values())
    max_value = max(uni_dict.values())
    uni_prob = uni_dict[word] / m
    lambda1 = (1 - uni_dict[word] / max_value) / 1000
    lambda2 = 1 - lambda1
    epsilon = 1000 / m
    return lambda1 * epsilon + lambda2 * uni_prob


def is_negative(pos_uni_dict, pos_bi_dict, neg_uni_dict, neg_bi_dict, string: str,
                pos_repetitive=None, neg_repetitive=None, mode=0):
    string = preprocess(list(string.split()))
    if pos_repetitive and neg_repetitive:
        temp_string = string
        for word in temp_string:
            if pos_repetitive.count(word) or neg_repetitive.count(word):
                string.remove(word)

    pos_uni_prob = unigram(pos_uni_dict, string[0])
    neg_uni_prob = unigram(neg_uni_dict, string[0])

    if mode == 0:
        pos_bi_prob = 1
        for word in range(len(string) - 1):
            pos_bi_prob *= bigram(pos_uni_dict, pos_bi_dict, string[word], string[word + 1])
        neg_bi_prob = 1
        for word in range(len(string) - 1):
            neg_bi_prob *= bigram(neg_uni_dict, neg_bi_dict, string[word], string[word + 1])
        if (neg_uni_prob / pos_uni_prob) * (neg_bi_prob / pos_bi_prob) > 1.25:
            return True
        else:
            return False
    else:
        pos_uni_probes = 1
        for word in string:
            pos_uni_probes *= unigram(pos_uni_dict, word)
        neg_uni_probes = 1
        for word in string:
            neg_uni_probes *= unigram(neg_uni_dict, word)
        if (neg_uni_prob / pos_uni_prob) * (neg_uni_probes / pos_uni_probes) > 1.25:
            return True
        else:
            return False


def get_string(pos_uni_dict, pos_bi_dict, neg_uni_dict, neg_bi_dict,
               pos_repetitive=None, neg_repetitive=None, mode=0):
    while True:
        input1 = input()
        if input1 == "!q":
            return True
        if is_negative(pos_uni_dict, pos_bi_dict, neg_uni_dict, neg_bi_dict, input1,
                       pos_repetitive, neg_repetitive, mode):
            print("filter this")
        else:
            print("not filter this")


def save_dict(my_dict, name):
    write = open("saved_data/"+name+".txt", "wb")
    dill.dump(my_dict, write)
    write.close()
    return True


def load_dict(name):
    read = open("saved_data/"+name+".txt", "rb")
    return dill.load(read)


def test(pos_uni_dict, pos_bi_dict, neg_uni_dict, neg_bi_dict, test_dataset,
         pos_repetitive=None, neg_repetitive=None, mode=0):
    filtered = 0
    not_filtered = 0
    for line in test_dataset:
        if is_negative(pos_uni_dict, pos_bi_dict, neg_uni_dict, neg_bi_dict, line,
                       pos_repetitive, neg_repetitive, mode):
            filtered += 1
        else:
            not_filtered += 1
    return filtered, not_filtered


if __name__ == '__main__':
    pos_dataset_read = open("rt-polarity.pos", "r")
    pos_dataset = pos_dataset_read.readlines()
    neg_dataset_read = open("rt-polarity.neg", "r")
    neg_dataset = neg_dataset_read.readlines()

    pos_dataset_train = []
    pos_dataset_test = []
    for i in range(int(0.95 * len(pos_dataset))):
        pos_dataset_train.append(pos_dataset[i])
    for i in range(int(0.95 * len(pos_dataset)), len(pos_dataset)):
        pos_dataset_test.append(pos_dataset[i])

    neg_dataset_train = []
    neg_dataset_test = []
    for i in range(int(0.95 * len(neg_dataset))):
        neg_dataset_train.append(neg_dataset[i])
    for i in range(int(0.95 * len(neg_dataset)), int(1 * len(neg_dataset))):
        neg_dataset_test.append(neg_dataset[i])

    pos_uni_dict1, pos_bi_dict1, pos_repetitive1 = make_dict(pos_dataset_train, 0)
    neg_uni_dict1, neg_bi_dict1, neg_repetitive1 = make_dict(neg_dataset_train, 0)

    get_string(pos_uni_dict1, pos_bi_dict1, neg_uni_dict1, neg_bi_dict1, pos_repetitive1, neg_repetitive1, 0)

    """save dictionaries to files"""
    # save_dict(pos_uni_dict1, "pos_uni_dict")
    # save_dict(neg_uni_dict1, "neg_uni_dict")
    # save_dict(pos_bi_dict1, "pos_bi_dict")
    # save_dict(neg_bi_dict1, "neg_bi_dict")
    # save_dict(pos_repetitive1, "pos_repetitive")
    # save_dict(neg_repetitive1, "neg_repetitive")

    """load dictionaries with files"""
    # pos_uni_dict1 = load_dict("pos_uni_dict")
    # pos_bi_dict1 = load_dict("pos_bi_dict")
    # pos_repetitive1 = load_dict("pos_repetitive")
    # neg_uni_dict1 = load_dict("neg_uni_dict")
    # neg_bi_dict1 = load_dict("neg_bi_dict")
    # neg_repetitive1 = load_dict("neg_repetitive")

    """test model"""
    # i1, j1 = test(pos_uni_dict1, pos_bi_dict1, neg_uni_dict1,
    #               neg_bi_dict1, neg_dataset_test, pos_repetitive1, neg_repetitive1, 0)
    # i2, j2 = test(pos_uni_dict1, pos_bi_dict1, neg_uni_dict1,
    #               neg_bi_dict1, pos_dataset_test, pos_repetitive1, neg_repetitive1, 0)
    # n_precision = i1 / (i1 + i2)
    # n_recall = i1 / (i1 + j1)
    # n_score = 2 * n_precision * n_recall / (n_precision + n_recall)
    # p_precision = j2 / (j1 + j2)
    # p_recall = j2 / (i2 + j2)
    # p_score = 2 * p_precision * p_recall / (p_precision + p_recall)
    # print("Negative Precision:", "{:.2f}".format(n_precision))
    # print("Negative Recall:", "{:.2f}".format(n_recall))
    # print("Negative Score:", "{:.2f}".format(n_score))
    # print("Positive Precision:", "{:.2f}".format(p_precision))
    # print("Positive Recall:", "{:.2f}".format(p_recall))
    # print("Positive Score:", "{:.2f}".format(p_score))
