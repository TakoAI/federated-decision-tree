# Federated Distribution Tree

# Dependencies
import pandas as pd
import numpy as np
from collections import Counter
from fbd import FBD

class FDTRoot:
    """
    Class for hosting the branches for a federated distribution tree
    """
    def __init__(
        self,
        depth: int,
        features: list,
        n: int,
        node_type: str,
        rule: str,
        algo_type: str = "classification", # classificatin | regression | bayes
        counts: Counter = None,
        yhat: float = None,
        ystd: float = None,
        fbd: FBD = None
    ):
        self.depth = depth
        self.features = features
        self.n = n
        self.node_type = node_type
        self.algo_type = algo_type
        self.rule = rule
        self.counts = counts
        self.yhat = yhat
        self.ystd = ystd
        self.fbd = fbd
        self.branches = None

    @staticmethod
    def Counter_dominatant(counts: Counter) -> int:
        """
        Sorting the counts and saving the final prediction of the node
        """
        counts_sorted = list(sorted(counts.items(), key=lambda item: item[1]))
        if len(counts_sorted) > 0:
            return counts_sorted[-1][0]
        return None

    def print_info(self, width = 4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const

        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info()

        if self.branches is not None:
            for branch in self.branches:
                branch.left.print_tree()
                branch.right.print_tree()

    def get_stats(self):
        """
        Return statistics of the federated distribution tree
        """
        if self.branches is not None:
            count = 0
            for branch in self.branches:
                stats_left = branch.left.get_stats()
                stats_right = branch.right.get_stats()
                count += stats_left["count"]
                count += stats_right["count"]
            return {"count": count}
        else:
            return {"count": 1}

    def predict(self, X: pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            if self.algo_type == "bayes":
                pred = self.predict_one(x)
                predictions.append(max(pred, key=pred.get))
            else:
                predictions.append(self.predict_obs(x))

        return predictions

    def predict_obs(self, values: pd.Series) -> int:
        """
        Method to predict the class given a set of features
        """
        if self.branches:
            node_map = {}
            node_arr = []
            for branch in self.branches:
                best_feature = branch.feature
                best_value = branch.value

                if (values[best_feature] < best_value):
                    y_pred = branch.left.predict_obs(values)
                else:
                    y_pred = branch.right.predict_obs(values)

                if self.algo_type == "classification":
                    if y_pred in node_map:
                        node_map[y_pred] += branch.n
                    else:
                        node_map[y_pred] = branch.n
                else:
                    node_arr.append(y_pred)

            if self.algo_type == "classification":
                return self.Counter_dominatant(Counter(node_map))
            else:
                return np.mean(node_arr)
        else:
            return self.yhat
        
    def predict_one(self, values: pd.Series) -> dict:
        """
        Method to predict the class vs expectation given a set of features
        """
        if self.branches:
            node_map = {}
            for branch in self.branches:
                best_feature = branch.feature
                best_value = branch.value
                
                if (values[best_feature] < best_value):
                    pred = branch.left.predict_one(values)
                else:
                    pred = branch.right.predict_one(values)
                
                for k in pred:
                    if k in node_map:
                        node_map[k] += pred[k]
                    else:
                        node_map[k] = pred[k]
            return node_map
        if self.fbd:
            return self.fbd.predict_one(values)
        return {}

    def merge(self, tree):
        """
        Merge another federated distribution tree to support federated learning
        """
        if self.algo_type == "classification":
            self.counts.update(tree.counts)
            self.yhat = self.Counter_dominatant(self.counts)
        elif self.algo_type == "regression":
            self.yhat = (self.n * self.yhat + tree.n * tree.yhat) / (self.n + tree.n)
        elif self.algo_type == "bayes":
            self.fbd.merge(tree.fbd)
        if self.branches and tree.branches:
            for t in tree.branches:
                no_match = True
                for s in self.branches:
                    if s.feature == t.feature and s.direct == t.direct:
                        if self.algo_type == "classification":
                            no_match = False
                            s.left.merge(t.left)
                            s.right.merge(t.right)
                            new_count = s.n + t.n
                            s.n = new_count
                        elif self.algo_type == "regression":
                            diff2_left = (s.left.yhat - t.left.yhat) ** 2
                            dstd2_left = s.left.ystd ** 2 + t.left.ystd ** 2
                            diff2_right = (s.right.yhat - t.right.yhat) ** 2
                            dstd2_right = s.right.ystd ** 2 + t.right.ystd ** 2
                            if diff2_left < dstd2_left and diff2_right < dstd2_right:
                                s.left.merge(t.left)
                                s.right.merge(t.right)
                                no_match = False
                                new_count = s.n + t.n
                                s.value = (s.n * s.value + t.n * t.value) / new_count
                                s.n = new_count
                        elif self.algo_type == "bayes":
                            no_match = False
                            s.left.merge(t.left)
                            s.right.merge(t.right)
                            new_count = s.n + t.n
                            s.value = (s.n * s.value + t.n * t.value) / new_count
                            s.n = new_count
                if no_match:
                    self.branches.append(t)
        elif tree.branches:
            self.branches = tree.branches

class FDTBranch:
    """
    Class for creating the decision for a federated distribution tree
    """
    def __init__(
        self,
        n: int,
        feature: str,
        value: float,
        left: FDTRoot,
        right: FDTRoot,
        direct: bool
    ):
        self.n = n
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.direct = direct

class FDT:
    """
    Class for creating the tree for a federated distribution tree
    """
    def __init__(
        self,
        min_samples_split = None,
        max_depth = None,
        depth = None,
        node_type = None,
        rule = None,
        algo_type: str = "classification"
    ):
        # Saving the hyper parameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        # Default current depth of node
        self.depth = depth if depth else 0

        # Type of node
        self.node_type = node_type if node_type else 'root'

        # Rule for spliting (previous stage split)
        self.rule = rule if rule else ""

        # Algorithm with classification and regression
        self.algo_type = algo_type

        # Initiating the branches with the structure [FDTBranch]
        self.branches = None

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list.
        """
        return np.convolve(x, np.ones(window), 'valid') / window

    @staticmethod
    def GINI_impurity(counts: Counter) -> float:
        """
        Given the observations of a multi class calculate the GINI impurity
        """
        # multi class calculation from Counter
        multi_count = [i[1] for i in counts.items()]
        n = sum(multi_count)
        gini = 1 - sum([(i / n) ** 2 for i in multi_count])
        
        # Returning the gini impurity
        return gini

    def get_GINI(self):
        """
        Function to calculate the GINI impurity of a node
        """
        # Getting the GINI impurity
        return self.GINI_impurity(self.counts)

    @staticmethod
    def MSE_loss(ytrue, yhat):
        """
        Return the mse comparing Y to the prediction
        """
        # Getting the residuals
        r = np.sum((ytrue - yhat) ** 2)

        # Getting the average and returning
        n = len(ytrue)
        
        if n == 0:
            return float('inf')
        return r / n

    def get_MSE(self):
        """
        Function to calculate the mean square error
        """
        return self.MSE_loss(self.Y, np.mean(self.Y))
    
    @staticmethod
    def FBD_loss(X, Y):
        tfbd = FBD().fit(X, Y)
        return FDT.GINI_impurity(Counter(tfbd.predict(X) == np.array(Y)))
    
    def get_FBDLoss(self):
        return self.GINI_impurity(Counter(self.fbd.predict(self.X) == np.array(self.Y)))
    
    def get_Loss(self):
        # GINI impurity
        if self.algo_type == "classification":
            return self.get_GINI()
        elif self.algo_type == "regression":
            return self.get_MSE()
        elif self.algo_type == "bayes":
            return self.get_FBDLoss()
        return 0.0

    def best_split(self) -> tuple:
        """
        Given the X features and Y targets calculates the best split
        """
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y
        
        # Getting the loss for the base input
        loss_base = self.get_Loss()

        # Finding which split yields the best GINI gain
        max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None
        best_direction = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Spliting the dataset
                left_df = Xdf[Xdf[feature]<value]
                left_x = left_df[self.features]
                left_y = left_df['Y']
                right_df = Xdf[Xdf[feature]>=value]
                right_x = right_df[self.features]
                right_y = right_df['Y']

                # Getting the obs count from the left and the right data splits
                n_left = len(left_y)
                n_right = len(right_y)

                if self.algo_type == "classification":
                    # Getting the left and right gini impurities
                    loss_left = self.GINI_impurity(Counter(left_y))
                    loss_right = self.GINI_impurity(Counter(right_y))
                elif self.algo_type == "regression":
                    # Getting the left and right mse loss
                    loss_left = self.MSE_loss(left_y, np.mean(left_y))
                    loss_right = self.MSE_loss(right_y, np.mean(right_y))
                elif self.algo_type == "bayes":
                    loss_left = self.FBD_loss(left_x, left_y)
                    loss_right = self.FBD_loss(right_x, right_y)

                # Calculating the weighted loss
                wloss_left = n_left / (n_left + n_right) * loss_left
                wloss_right = n_right / (n_left + n_right) * loss_right
                wloss = wloss_left + wloss_right

                # Calculating the gain
                gain = loss_base - wloss

                # Checking if this is the best split so far
                if gain > max_gain:
                    best_feature = feature
                    best_value = value
                    best_direction = bool(wloss_left > wloss_right)

                    # Setting the best gain to the current one
                    max_gain = gain

        return (best_feature, best_value, best_direction)
    
    def init_paras(
        self,
        X: pd.DataFrame,
        Y: list
    ):
        """
        Initialize the parameters of the tree
        """
        # Saving the data to the node
        self.X = X
        self.Y = Y

        # Extracting all the features
        self.features = list(self.X.columns)

        # Init None variables
        self.counts = None
        self.yhat = None
        self.ystd = None
        self.fbd = None
        
        # Calculating the counts of Y in the node
        if self.algo_type == "classification":
            self.counts = Counter(Y)
            self.yhat = FDTRoot.Counter_dominatant(self.counts)
        
        # Federated bayes decision
        if self.algo_type == "bayes":
            self.fbd = FBD().fit(X, Y)

        # Regression
        if self.algo_type == "regression":
            self.yhat = np.mean(Y)
            self.ystd = np.std(Y)

        # Saving the number of observations in the node
        self.n = len(Y)

    def fit(
        self,
        X: pd.DataFrame,
        Y: list
    ) -> FDTRoot:
        """
        Recursive method to create the federated distribution tree
        """
        self.init_paras(X, Y)
        
        # Making a df from the data
        df = self.X.copy()
        df['Y'] = self.Y
        result = FDTRoot(
            depth = self.depth,
            features = self.features,
            n = self.n,
            node_type = self.node_type,
            rule = self.rule,
            algo_type = self.algo_type,
            counts = self.counts,
            yhat = self.yhat,
            ystd = self.ystd,
            fbd = self.fbd
        )

        # If there is GINI to be gained, we split further
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            # Getting the best split
            best_feature, best_value, best_direction = self.best_split()

            if best_feature is not None:
                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()
                left_type = ["sub_node", "main_node"][best_direction]
                right_type = ["main_node", "sub_node"][best_direction]

                # Creating the left and right nodes
                left = FDT(
                    depth = self.depth + 1,
                    max_depth = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    node_type = left_type,
                    rule = f"{best_feature} <= {round(best_value, 3)}",
                    algo_type = self.algo_type
                ).fit(
                    left_df[self.features],
                    left_df['Y'].values.tolist()
                )

                right = FDT(
                    depth = self.depth + 1,
                    max_depth = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    node_type = right_type,
                    rule = f"{best_feature} > {round(best_value, 3)}",
                    algo_type = self.algo_type
                ).fit(
                    right_df[self.features],
                    right_df['Y'].values.tolist()
                )

                # Create the FDT branch
                result.branches = [FDTBranch(
                    n = 1,
                    feature = best_feature,
                    value = best_value,
                    left = left,
                    right = right,
                    direct = best_direction
                )]
        return result
