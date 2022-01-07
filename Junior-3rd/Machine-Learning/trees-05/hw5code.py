import numpy as np  
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):    
    zip_vector = sorted(list(zip(feature_vector, target_vector)), key=lambda x: x[0])
    feat, target = zip(*zip_vector) 
    
    Rl = np.arange(1, len(target), 1)
    R = feature_vector.shape[0]
        
    sl_1 = np.cumsum(target[:-1])
    pl_1 = sl_1 / Rl

    sl_0 = Rl - sl_1
    pl_0 = sl_0 / Rl
    
    sr_1 = np.cumsum(target[:0:-1])
    pr_1 = (sr_1 / Rl)[::-1]

    sr_0 = Rl - sr_1
    pr_0 = (sr_0 / Rl)[::-1]
    
    H_Rl = 1 - pl_1 ** 2 - pl_0 ** 2                 
    H_Rr = 1 - pr_1 ** 2 - pr_0 ** 2
    
    unique_feat, index, counts = np.unique(feat, return_index=True, return_counts=True)
    ginis = (- (Rl / R) * H_Rl - (Rl[::-1] / R) * H_Rr)[(counts + index - 1)[:-1]]
    gini_best = np.max(ginis)
    
    thresholds = (unique_feat[1:] + unique_feat[:-1]) / 2
    threshold_best = thresholds[np.argmax(ginis)]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])   # ['a': 5, 'b': 3, 'c': 2]
                clicks = Counter(sub_X[sub_y == 1, feature])  # ['a': 2, 'b': 1, 'c': 0]
                ratio = {}
                for key, current_count in counts.items():
                    #if key in clicks:
                    current_click = clicks[key]   # 2 = clicks[a]
                    #else:
                    #    current_click = 0  # 0 = clicks[c]
                    ratio[key] = current_click / current_count  # ratio['a'] = 2/5
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))  # ['a', 'c', 'b']
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories))))) # ['a' : 0, 'c' : 1, 'b' : 2]

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))  # [1, 2, 2, 1, 0, 2]
            else:
                raise ValueError

            if len(np.unique(feature_vector)) in [0, 1]:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
            node["categories_split"] = None
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
            node["threshold"] = None
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == 'nonterminal':
            split_feat = node["feature_split"]
            if self.feature_types[split_feat] == 'real' and x[split_feat] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            if self.feature_types[split_feat] == 'categorical' and x[split_feat] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        return node["class"]

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

