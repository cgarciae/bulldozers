from sklearn import preprocessing as sk_preprocessing


class LabelEncoder(sk_preprocessing.LabelEncoder):
    def fit(self, X, y=None):
        return super().fit(X)

    def fit_transform(self, X, y=None):
        return super().fit_transform(X[:, 0])
