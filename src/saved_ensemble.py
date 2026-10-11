"""Equal-weight probability averaging of already fitted subset forests."""
import numpy as np


class SavedSubsetEnsemble:
    def __init__(self, members, union_columns):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = len(union_columns)
        lookup = {int(column): i for i, column in enumerate(union_columns)}
        self.members = []
        self.feature_importances_ = np.zeros(len(union_columns))
        for saved in members:
            model = saved['model']
            if list(model.classes_) != [0, 1]:
                raise ValueError('Members must have binary classes [0, 1].')
            positions = np.array([lookup[int(c)] for c in saved['columns']], dtype=int)
            self.members.append((model, positions))
            self.feature_importances_[positions] += model.feature_importances_ / len(members)
        if not self.members:
            raise ValueError('At least one fitted model is required.')

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('Input must contain the ensemble union columns in saved order.')
        total = np.zeros((len(X), 2), dtype=float)
        for model, positions in self.members:
            total += model.predict_proba(X[:, positions])
        return total / len(self.members)
