import csv
from scipy.stats import chisquare
import numpy as np1
import pandas as pd1
import argparse
import pickle as pkl1

# Fixed set of unique features to 5 according to provided dataset
unique_features = 5

# TreeNode is the node in Decision Tree
class TreeNode:
    def __init__(self, data='T', children=None):
        if children is None:
            children = [-100] * unique_features
        self.nodes = list(children)
        self.data = data

count_nodes = 0

def save_tree(TreeNode, filename):
    with open(filename, 'w') as obj:
        pkl1.dump(TreeNode, obj)

# Getting the P value for particular feature
def getPValue(unique_features, features, pcount, ncount):

    efreq = []
    ofreq = []
    total = pcount + ncount

    it = 0
    while(it < len(unique_features)):

        ndata = features[features['value'] == unique_features[it]]

        total_count = ndata.shape[0]

        opcount = float(ndata['result'].sum())
        oncount = total_count - opcount

        p_probability = float(pcount) / float(total)
        n_probability = float(ncount) / float(total)

        ofreq = ofreq + [opcount, oncount]

        it += 1

        epcount = float(p_probability) * total_count
        encount = float(n_probability) * total_count

        efreq = efreq + [epcount, encount]

    #Passing observed frequencies and expected frequencies in chi-square function to get p value
    c, p_val = chisquare(ofreq, efreq)
    return p_val

def chiSquareResult(train_class, train_data):

    features = pd1.DataFrame()
    features['value'] = train_data
    features['result'] = train_class

    total = len(train_class)
    pcount = (train_class[0] == 1).sum()
    ncount = total - pcount

    unique_feats = train_data.unique()

    p_val = getPValue(unique_feats, features, pcount, ncount)
    return p_val


def minEntropyFeat(train_class, train_data, feature_val):

    best_entropy = float('inf')
    length = len(feature_val)
    it = 0
    while(it < length):

        entropy = 0.0
        features = pd1.DataFrame()
        features['feat_values'] = train_data[it]
        features['output'] = train_class

        features_unique = train_data[it].unique()

        it1 = 0
        while(it1 < len(features_unique)):

            value = features_unique[it1]

            ndata = features[train_data[it] == value]
            total_count = ndata.shape[0]

            pos_count = float(ndata["output"].sum())
            neg_count = total_count - pos_count

            prob_feature = float(ndata.shape[0]) / features.shape[0]

            prob_pos = pos_count / total_count
            prob_neg = neg_count / total_count

            value = 0
            if prob_pos != 0:
                value = prob_pos * np1.log2(prob_pos)
            if prob_neg != 0:
                value += prob_neg * np1.log2(prob_neg)

            feature_entropy = -1 * (value)
            entropy += prob_feature * feature_entropy

            it1 += 1

        if entropy < best_entropy:
            best_entropy = entropy
            best_feat = it
        it += 1
    return best_feat

# Decision Tree ID3 algorithm implementation
def decisionTree(train_class, train_data, feature_data, p_threshold):

    positive_count = len(train_class[train_class[0] == 1])
    negative_count = len(train_class[train_class[0] == 0])
    total_count = len(train_class)
    feat_length = len(feature_data)

    global count_nodes
    count_nodes += 1

    if feat_length == 0:
        return TreeNode('T') if positive_count >= negative_count else TreeNode('F')

    if negative_count == total_count:
        return TreeNode('F')

    if positive_count == total_count:
        return TreeNode('T')

    max_gain_feat = minEntropyFeat(train_class, train_data, feature_data)
    p_value = chiSquareResult(train_class, train_data[max_gain_feat])

    if p_value <= p_threshold:

        root = TreeNode(max_gain_feat + 1)
        new_features = list(feature_data)

        #Check if feature is in my list
        if(max_gain_feat not in new_features):
            return TreeNode('T') if (positive_count >= negative_count) else TreeNode('F')
        else:
            new_features.remove(max_gain_feat)

        it = 1
        while it <= unique_features:

            if it not in train_data:
                abDecision = 'T' if positive_count > negative_count else 'F'
                root.nodes[it - 1] = abDecision
                it += 1
                continue

            # If there are no points with 1 labels in sample space, then return False as decision,
            if negative_count == 0:
                root.nodes[it-1] = TreeNode('T')
                it += 1
                continue
            elif positive_count == 0:
                root.nodes[it - 1] = TreeNode('F')
                it += 1
                continue

            t_data = pd1.DataFrame(train_data)
            t_data['class'] = train_class[0]
            t_data = train_data[train_data[max_gain_feat] == it]

            train_class_new = pd1.DataFrame()
            train_class_new[0] = t_data['class']
            root.nodes[it - 1] = decisionTree(train_class_new, t_data, new_features, p_threshold)
            it += 1
    else:
        return TreeNode('T') if (positive_count >= negative_count) else TreeNode('F')
    return root

# Testing on test data
def computeTestData(root, sample):

    if root.data == 'T':
        return 1
    if root.data == 'F':
        return 0
    val = int(sample[int(root.data) - 1]) - 1
    root = root.nodes[val]
    return computeTestData(root, sample)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True)
    parser.add_argument('-f1', required=True)
    parser.add_argument('-f2', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-t', required=True)

    args = vars(parser.parse_args())
    p_value = float(args['p'])
    train_output = "train_label.csv"
    train = args['f1']
    test = args['f2']
    output = args['o']
    tree = args['t']

    data_features = pd1.read_csv('featnames.csv', header=None, delim_whitespace=True)

    features = []
    ufeatures = len(data_features)
    it = 0
    while (it < ufeatures):
        features.append(it)
        it += 1

    train_data = pd1.read_csv(train, header=None, delim_whitespace=True)
    train_class = pd1.read_csv(train_output, header=None)

    print "Training Started"
    root = decisionTree(train_class, train_data, features, p_value)
    save_tree(root, tree)
    print "Training Completed"

    print "Number of nodes generated ", count_nodes

    print "Testing Started"
    result = []

    test_data = pd1.read_csv(test, header=None, delim_whitespace=True)

    # generate random labels
    for i in range(len(test_data)):
        val = computeTestData(root, test_data[i:i + 1])
        result.append([val])

    print "Testing Completed"

    with open(output, "wb") as f:
        worker = csv.writer(f)
        worker.writerows(result)
