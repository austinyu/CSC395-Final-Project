from sklearn.metrics import ConfusionMatrixDisplay
from data_structs import NewsDataLoader, ScorePackage, EvaluationDataLoader

def model_predict(
    model: any,
    name_model: str,
    news_loader: NewsDataLoader
) -> EvaluationDataLoader:
    """
    Performs predictions using a trained model on the given news_loader.

    Parameters:
    - model (RandomForestClassifier): The trained model for prediction.
    - name_model (str): The name of the model.
    - news_loader (NewsDataLoader): An instance of NewsDataLoader containing the predictors and targets.

    Returns:
    - evaluation_loader (EvaluationDataLoader): An instance of EvaluationDataLoader containing the predicted and actual targets.
    """
    # Unpack loaders
    news_predictors, news_target = news_loader.predictors, news_loader.target

    # Make predictions on test data
    news_target_predicted = model.predict(news_predictors)

    evaluation_loader = EvaluationDataLoader(
        name_dataset='train',
        name_model=name_model,
        target_predicted=news_target_predicted,
        target_actual=news_target
    )

    return evaluation_loader


def model_evaluate(
    evaluation_loader: EvaluationDataLoader,
    name_dataset: str,
    display_confusion: bool = True
) -> ScorePackage:
    """
    Evaluates the performance of a model using the provided EvaluationDataLoader.

    Parameters:
    - evaluation_loader (EvaluationDataLoader): An instance of EvaluationDataLoader containing the predicted and actual targets.
    - name_dataset (str): The name of the dataset being evaluated.
    - display_confusion (bool): Whether to display the confusion matrix. Default is True.

    Returns:
    - score_package (ScorePackage): An instance of ScorePackage containing evaluation scores.
    """
    name_model = evaluation_loader.name_model

    if display_confusion:
        ConfusionMatrixDisplay.from_predictions(
            evaluation_loader.target_actual,
            evaluation_loader.target_predicted
        )

    score_package = ScorePackage(
        name_dataset=name_dataset,
        name_model=name_model
    )
    score_package.construct_eval(evaluation_loader)

    score_package.pretty_print()
    return score_package


def model_predict_evaluate(
    model: any,
    name_model: str,
    name_dataset: str,
    news_loader: NewsDataLoader,
    display_confusion: bool = True
) -> ScorePackage:
    """
    Performs predictions and evaluation on a given dataset using a trained model.

    Parameters:
    - model (RandomForestClassifier): The trained model for prediction and evaluation.
    - name_model (str): The name of the model.
    - name_dataset (str): The name of the dataset being evaluated.
    - news_loader (NewsDataLoader): An instance of NewsDataLoader containing the predictors and targets.
    - display_confusion (bool): Whether to display the confusion matrix. Default is True.

    Returns:
    - score_package (ScorePackage): An instance of ScorePackage containing evaluation scores.
    """
    score_package = model_evaluate(
        model_predict(
            model=model,
            name_model=name_model,
            news_loader=news_loader
        ),
        name_dataset,
        display_confusion
    )

    return score_package