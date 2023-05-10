from sklearn.metrics import ConfusionMatrixDisplay
from data_structs import NewsDataLoader, ScorePackage, EvaluationDataLoader

from sklearn.metrics import ConfusionMatrixDisplay

def model_predict(model, name_model, train_news_loader: NewsDataLoader, test_news_loader: NewsDataLoader):
    """
    Perform predictions using a machine learning model on the training and test datasets.

    Parameters:
        model (object): The machine learning model used for prediction.
        name_model (str): The name of the model.
        train_news_loader (NewsDataLoader): An instance of the NewsDataLoader class containing the training data.
        test_news_loader (NewsDataLoader): An instance of the NewsDataLoader class containing the test data.

    Returns:
        Tuple: A tuple containing the instances of EvaluationDataLoader for the training and test datasets.

    Description:
        This function takes a machine learning model, along with the training and test data loaders, and performs 
        predictions using the model. It unpacks the predictors and target variables from the loaders, makes 
        predictions on the test data, and creates instances of EvaluationDataLoader for the training and test 
        datasets. These instances contain the predicted and actual target values, which are returned for 
        further evaluation.
    """
    # unpack loaders
    train_news_predictors, train_news_target = train_news_loader.predictors, train_news_loader.target
    test_news_predictors, test_news_target = test_news_loader.predictors, test_news_loader.target
    # Make predictions on test data
    train_news_target_predicted = model.predict(train_news_predictors)
    test_news_target_predicted = model.predict(test_news_predictors)
    train_evaluation_loader = EvaluationDataLoader(
        name_dataset='train',
        name_model=name_model,
        target_predicted=train_news_target_predicted,
        target_actual=train_news_target)
    test_evaluation_loader = EvaluationDataLoader(
        name_dataset='test',
        name_model=name_model,
        target_predicted=test_news_target_predicted,
        target_actual=test_news_target)
    return (train_evaluation_loader, test_evaluation_loader)

def model_evaluate(train_evaluation_loader: EvaluationDataLoader,
                   test_evaluation_loader: EvaluationDataLoader,
                   display_confusion = True):
    """
    Evaluate the performance of a model based on the predicted and actual target values.

    Parameters:
        train_evaluation_loader (EvaluationDataLoader): An instance of EvaluationDataLoader containing 
            the predicted and actual target values for the training dataset.
        test_evaluation_loader (EvaluationDataLoader): An instance of EvaluationDataLoader containing 
            the predicted and actual target values for the test dataset.
        display_confusion (boolean): switch for displaying the confusion matrix.

    Returns:
        Tuple: A tuple containing the instances of ScorePackage for the training and test datasets.

    Description:
        This function evaluates the performance of a model based on the predicted and actual target values. 
        It takes instances of EvaluationDataLoader for the training and test datasets as input. The function 
        displays the confusion matrix for the test dataset using ConfusionMatrixDisplay.from_predictions, 
        constructs instances of ScorePackage based on the evaluation results, prints the evaluation scores, 
        and returns the instances of ScorePackage containing the evaluation scores.
    """
    name_model = train_evaluation_loader.name_model
    if display_confusion:
        ConfusionMatrixDisplay.from_predictions(
            test_evaluation_loader.target_actual,
            test_evaluation_loader.target_predicted)

    train_score_package = ScorePackage(
        name_dataset='train', name_model=name_model)
    train_score_package.construct_eval(train_evaluation_loader)
    test_score_package = ScorePackage(
        name_dataset='test', name_model=name_model)
    test_score_package.construct_eval(test_evaluation_loader)

    train_score_package.pretty_print()
    test_score_package.pretty_print()
    return (train_score_package, test_score_package)

def model_predict_evaluate(model, name_model, 
                           train_news_loader: NewsDataLoader, 
                           test_news_loader: NewsDataLoader,
                           display_confusion = True):
    """
    Combine the prediction and evaluation steps for a machine learning model.

    Parameters:
        model (object): The machine learning model used for prediction.
        name_model (str): The name of the model.
        train_news_loader (NewsDataLoader): An instance of NewsDataLoader containing the training data.
        test_news_loader (NewsDataLoader): An instance of NewsDataLoader containing the test data.

    Returns:
        Tuple: A tuple containing the instances of ScorePackage for the training and test datasets.

    Description:
        This function combines the prediction and evaluation steps for a model. It takes a machine 
        learning model, along with the training and test data loaders, and calls the model_predict 
        function to obtain the predicted and actual target values. It then calls the model_evaluate 
        function to evaluate the model's performance based on these values. Finally, it returns the 
        instances of ScorePackage containing the evaluation scores for the training and test datasets.
    """
    score_package_train, score_package_test = model_evaluate(
        *model_predict(
            model=model,
            name_model=name_model,
            train_news_loader=train_news_loader,
            test_news_loader=test_news_loader), 
        display_confusion
    )
    return score_package_train, score_package_test
