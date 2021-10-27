from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

class SearchParams(object):
    def __init__(self, X_train, y_train, model, hyperparameters):

        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters

    def Kfold(self):
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def RandomSearch(self):
        cv = self.Kfold()
        clf = RandomizedSearchCV(self.model,
                                 self.hyperparameters,
                                 random_state=42,
                                 n_iter=100,
                                 cv=cv,
                                 verbose=0,
                                 n_jobs=-1)

        best_model = clf.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_)

        print("best: %f using %s" % (message))

        return best_model, best_model.best_params_

    def GridSearch(self):
        cv = self.Kfold()
        clf = GridSearchCV(self.model,
                           self.hyperparameters,
                           cv=cv,
                           verbose=0,
                           n_jobs=-1)
        best_model = clf.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_)

        print("best score: %f using %s" % (message))

        return best_model, best_model.best_params_

    def BestModelPredict(self, X_test, random=False):
        if random:
            best_model, _ = self.RandomSearch()
            pred = best_model.predict(X_test)
        else:
            best_model, _ = self.GridSearch()
            pred = best_model.predict(X_test)
        return pred
