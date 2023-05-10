from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from typing import Any, Optional, Dict

class NewsDataLoader:
    def __init__(self, predictors: Any, target: Any) -> None:
        self.predictors = predictors
        self.target = target

class EvaluationDataLoader:
    def __init__(self, name_dataset: str, name_model: str,
                 target_actual: Any, target_predicted: Any) -> None:
        self.name_dataset = name_dataset
        self.name_model = name_model
        self.target_actual = target_actual
        self.target_predicted = target_predicted

class ScorePackage:
    def __init__(self, name_dataset: str, name_model: str, scores: Optional[Dict[str, Any]] = None) -> None:
        self.package = dict()
        self.name_dataset = name_dataset
        self.name_model = name_model
        
        if scores is None:
            return
        
        self.accuracy = round(scores['test_accuracy'].mean(), 3)
        self.precision = round(scores['test_precision'].mean(), 3)
        self.recall = round(scores['test_recall'].mean(), 3)
        self.f1 = round(scores['test_f1'].mean(), 3)

    def update_package(self, accuracy: Optional[float] = None, precision: Optional[float] = None,
                       recall: Optional[float] = None, f1: Optional[float] = None) -> None:
        if accuracy is not None:
            self.accuracy = round(accuracy, 3)
        if precision is not None:
            self.precision = round(precision, 3)
        if recall is not None:
            self.recall = round(recall, 3)
        if f1 is not None:
            self.f1 = round(f1, 3)

    def construct_eval(self, evaluation_loader: EvaluationDataLoader) -> None:
        target_actual = evaluation_loader.target_actual
        target_predicted = evaluation_loader.target_predicted

        test_accuracy = accuracy_score(target_actual, target_predicted)
        test_recall = recall_score(target_actual, target_predicted)
        test_precision = precision_score(target_actual, target_predicted)
        test_f1 = f1_score(target_actual, target_predicted, average='macro')
        
        self.update_package(test_accuracy, test_recall, test_precision, test_f1)

    def __str__(self) -> str:
        return f'Accuracy: {self.accuracy}, Precision: {self.precision}, Recall: {self.recall}, f1 score: {self.f1}'

    def pretty_print(self) -> None:
        print(f'For <{self.name_model}> evaluated on <{self.name_dataset}> Dataset:')
        print(f'    - accuracy: {self.accuracy}')
        print(f'    - precisionn: {self.precision}')
        print(f'    - recall: {self.recall}')
        print(f'    - f1 score: {self.f1}')
        
class ModelEnum:
    knn = 'knn'
    logisticRegression = 'logisticRegression'
    linearSVM = 'linearSVM'
    randomForest = 'randomForest'
    voting = 'voting'
    stack = 'stack'
    adaBoost = 'adaBoost'
    rnn = 'rnn'