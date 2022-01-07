from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample


sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)
            self.train_loss = np.full(self.early_stopping_rounds, np.inf)

        
        self.plot: bool = plot
        self.history = pd.DataFrame(data=None, columns=['train', 'valid'])
        

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        tr_idx = resample(np.arange(x.shape[0]), n_samples = int(self.subsample * x.shape[0]))
        offset = self.loss_derivative(y[tr_idx], predictions[tr_idx])
        
        N_model = self.base_model_class(**self.base_model_params).fit(x[tr_idx], -offset)
        new_pred = N_model.predict(x)
        
        self.models.append(N_model)
        self.gammas.append(self.find_optimal_gamma(y, predictions, new_pred) * self.learning_rate)
        

    def fit(self, x_train, y_train, x_valid, y_valid):
        
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        
        for i in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)
            self.history.loc[i] = [self.score(x_train, y_train), self.score(x_valid, y_valid)]
            
            if self.early_stopping_rounds is not None:
                np.append(self.train_loss, self.loss_fn(y_train, train_predictions))
                np.append(self.validation_loss, self.loss_fn(y_valid, valid_predictions))
                if self.early_stopping_rounds <= i:
                    if np.argmin(self.validation_loss) == len(self.validation_loss) - 1:
                        break
        

        if self.plot:
            if early_stopping_rounds is not None:
                plt.plot(self.train_loss)
                plt.plot(self.validation_loss)
            self.history.plot()
                
            

    def predict_proba(self, x):
        preds = np.sum(model.predict(x) * gamma for gamma, model in zip(self.gammas, self.models))
        probs = self.sigmoid(preds)
        return np.transpose(np.vstack([1 - probs, probs]))
            

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        return np.mean([obj.feature_importances_ for obj in self.models], axis=0)
