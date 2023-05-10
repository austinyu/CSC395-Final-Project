from sklearn.metrics import ConfusionMatrixDisplay
from data_structs import NewsDataLoader, ScorePackage, EvaluationDataLoader

from sklearn.metrics import ConfusionMatrixDisplay

def model_predict(model, name_model, news_loader: NewsDataLoader):
    # unpack loaders
    news_predictors, news_target = news_loader.predictors, news_loader.target
    # Make predictions on test data
    news_target_predicted = model.predict(news_predictors)
    evaluation_loader = EvaluationDataLoader(
        name_dataset='train',
        name_model=name_model,
        target_predicted=news_target_predicted,
        target_actual=news_target)
    return evaluation_loader

def model_evaluate(evaluation_loader: EvaluationDataLoader, 
                   name_dataset,
                   display_confusion = True):
    name_model = evaluation_loader.name_model
    if display_confusion:
        ConfusionMatrixDisplay.from_predictions(
            evaluation_loader.target_actual,
            evaluation_loader.target_predicted)

    score_package = ScorePackage(
        name_dataset=name_dataset, 
        name_model=name_model)
    score_package.construct_eval(evaluation_loader)

    score_package.pretty_print()
    return score_package

def model_predict_evaluate(model, 
                           name_model, 
                           name_dataset,
                           news_loader: NewsDataLoader, 
                           display_confusion = True):
    score_package = model_evaluate(
        model_predict(
            model=model,
            name_model=name_model,
            news_loader=news_loader),
        name_dataset, 
        display_confusion
    )
    return score_package
