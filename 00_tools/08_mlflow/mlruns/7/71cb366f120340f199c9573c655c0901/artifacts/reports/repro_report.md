# MLflow Repro Report

## Experiments

### 08_mlflow_artifacts

| run_id                           | status   | metrics.accuracy   | metrics.auc   | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|:-------------------|:--------------|:-------------------|:---------------|:----------------------|:-------------------|
| 698c86ffa18544ae84b40df32e6e3e53 | FINISHED |                    |               |                    |                |                       |                    |
| 491c09ecf9b9461f9d05253306714469 | FINISHED |                    |               |                    |                |                       |                    |

### 08_mlflow_basics

| run_id                           | status   |   metrics.accuracy |   metrics.auc | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|-------------------:|--------------:|:-------------------|:---------------|:----------------------|:-------------------|
| 77dd0c35e70a4bd5a4393ee9f6bddefc | FINISHED |             0.9632 |        0.9881 |                    | logreg         |                       |                    |
| 627c27522c84408da5da40f2f50e8e3f | FINISHED |             0.9632 |        0.9881 |                    | logreg         |                       |                    |

### 08_mlflow_experiment_compare

| run_id                           | status   |   metrics.accuracy | metrics.auc   |   metrics.f1_macro | params.model   |   params.n_estimators |   params.max_depth |
|:---------------------------------|:---------|-------------------:|:--------------|-------------------:|:---------------|----------------------:|-------------------:|
| 73e87b66c5b540829cd0641d842d5d12 | FINISHED |                  1 |               |                  1 |                |                   400 |                 -1 |
| 461a4d9f29544de0b12057bf01c938f1 | FINISHED |                  1 |               |                  1 |                |                   300 |                  8 |
| 5976d029581e43f1953e8635316f5153 | FINISHED |                  1 |               |                  1 |                |                   200 |                  6 |
| bef73e60f04a43e095245f0acfd6fac6 | FINISHED |                  1 |               |                  1 |                |                   100 |                 -1 |
| d111e4c640dd42e484bb512fd5e60708 | FINISHED |                  1 |               |                  1 |                |                   400 |                 -1 |

### 08_mlflow_model_logging

| run_id                           | status   |   metrics.accuracy |   metrics.auc | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|-------------------:|--------------:|:-------------------|:---------------|:----------------------|:-------------------|
| 08889eb541664d0b88d5be5f3a862ab5 | FINISHED |           0.982456 |       0.99537 |                    | logreg         |                       |                    |
| 46788901b54d4fc7b1fdb58ae8eb4d6d | FINISHED |           0.982456 |       0.99537 |                    | logreg         |                       |                    |

### 08_mlflow_registry

| run_id                           | status   | metrics.accuracy   | metrics.auc   | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|:-------------------|:--------------|:-------------------|:---------------|:----------------------|:-------------------|
| b4d3e118547141ab9cf925c4f9ec2399 | FINISHED |                    |               |                    | logreg         |                       |                    |

### 08_mlflow_xgboost

| run_id                           | status   |   metrics.accuracy |   metrics.auc | metrics.f1_macro   | params.model   |   params.n_estimators |   params.max_depth |
|:---------------------------------|:---------|-------------------:|--------------:|:-------------------|:---------------|----------------------:|-------------------:|
| 7ae57a0be5d84f4f917b48c7c373e080 | FINISHED |            0.95614 |      0.994709 |                    |                |                   300 |                  4 |
