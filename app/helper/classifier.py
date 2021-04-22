from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

SVM = 'SVM'
LOGISTIC_REGRESSION = "LOG"


class Classifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def classify(self, embedding_train, train_label):
        if self.classifier == SVM:
            svmmodel = SVC(kernel='linear', C=1)
            svmmodel.fit(embedding_train, train_label)
            return svmmodel
        elif self.classifier == LOGISTIC_REGRESSION:
            logit = LogisticRegression(C=5e1, solver='lbfgs',
                                       multi_class='ovr', random_state=5, max_iter=1000)
            logit.fit(embedding_train, train_label)
            return logit

class Evaluation:
    def evaluate(self, model, embedding_test, test_label):
        y_pred= model.predict(embedding_test)
        y_prob_bert_words_svm = model.decision_function(embedding_test)

        acc = accuracy_score(test_label, y_pred)
        # Result
        print("Accuracy: {:.2f}".format(acc*100), end='\n\n')
        cm = confusion_matrix(test_label, y_pred)
        print('Confusion Matrix:\n', cm)
        print(classification_report(test_label, y_pred))
