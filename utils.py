from sklearn.metrics import ConfusionMatrixDisplay
from data_structs import NewsDataLoader, ScorePackage, EvaluationDataLoader


from sklearn.metrics import ConfusionMatrixDisplay


def model_predict(model, name_model, train_news_loader: NewsDataLoader, test_news_loader: NewsDataLoader):
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
                   test_evaluation_loader: EvaluationDataLoader):
    name_model = train_evaluation_loader.name_model
    # TestConfusionn matrix
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


def model_predict_evaluate(model, name_model, train_news_loader: NewsDataLoader, test_news_loader: NewsDataLoader):
    score_package_train, score_package_test = model_evaluate(
        *model_predict(
            model=model,
            name_model=name_model,
            train_news_loader=train_news_loader,
            test_news_loader=test_news_loader)
    )
    return score_package_train, score_package_test
