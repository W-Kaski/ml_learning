# MLflow Repro Report

## Experiments

### 08_mlflow_artifacts

| run_id                           | status   | metrics.accuracy   | metrics.auc   | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|:-------------------|:--------------|:-------------------|:---------------|:----------------------|:-------------------|
| 92c7dd2e28fc4fa8a90063366fbf51b7 | FINISHED |                    |               |                    |                |                       |                    |
| 698c86ffa18544ae84b40df32e6e3e53 | FINISHED |                    |               |                    |                |                       |                    |
| 491c09ecf9b9461f9d05253306714469 | FINISHED |                    |               |                    |                |                       |                    |

### 08_mlflow_basics

| run_id                           | status   |   metrics.accuracy |   metrics.auc | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|-------------------:|--------------:|:-------------------|:---------------|:----------------------|:-------------------|
| 4ce88218d77e44e19c8f3fdbd7a40f19 | FINISHED |             0.9632 |        0.9881 |                    | logreg         |                       |                    |
| 77dd0c35e70a4bd5a4393ee9f6bddefc | FINISHED |             0.9632 |        0.9881 |                    | logreg         |                       |                    |
| 627c27522c84408da5da40f2f50e8e3f | FINISHED |             0.9632 |        0.9881 |                    | logreg         |                       |                    |

### 08_mlflow_experiment_compare

| run_id                           | status   |   metrics.accuracy | metrics.auc   |   metrics.f1_macro | params.model   |   params.n_estimators |   params.max_depth |
|:---------------------------------|:---------|-------------------:|:--------------|-------------------:|:---------------|----------------------:|-------------------:|
| b6ddf3a9df134bb9a91e19a86ab3bf23 | FINISHED |                  1 |               |                  1 |                |                   400 |                 -1 |
| dc59bc9528d74e96b9ecc0eba466b710 | FINISHED |                  1 |               |                  1 |                |                   300 |                  8 |
| 2f1b6cb94ffb41c289697df43bbe0ef3 | FINISHED |                  1 |               |                  1 |                |                   200 |                  6 |
| e933a3d5ad7a4a3d9430add8950be205 | FINISHED |                  1 |               |                  1 |                |                   100 |                 -1 |
| 73e87b66c5b540829cd0641d842d5d12 | FINISHED |                  1 |               |                  1 |                |                   400 |                 -1 |

### 08_mlflow_model_logging

| run_id                           | status   |   metrics.accuracy |   metrics.auc | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|-------------------:|--------------:|:-------------------|:---------------|:----------------------|:-------------------|
| 89a8740727df4bbf94aef47895d301b5 | FINISHED |           0.982456 |       0.99537 |                    | logreg         |                       |                    |
| 08889eb541664d0b88d5be5f3a862ab5 | FINISHED |           0.982456 |       0.99537 |                    | logreg         |                       |                    |
| 46788901b54d4fc7b1fdb58ae8eb4d6d | FINISHED |           0.982456 |       0.99537 |                    | logreg         |                       |                    |

### 08_mlflow_registry

| run_id                           | status   | metrics.accuracy   | metrics.auc   | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|:-------------------|:--------------|:-------------------|:---------------|:----------------------|:-------------------|
| 7782f936128d49438c319b8b420ae3e0 | FINISHED |                    |               |                    | logreg         |                       |                    |
| b4d3e118547141ab9cf925c4f9ec2399 | FINISHED |                    |               |                    | logreg         |                       |                    |

### 08_mlflow_repro_report

| run_id                           | status   | metrics.accuracy   | metrics.auc   | metrics.f1_macro   | params.model   | params.n_estimators   | params.max_depth   |
|:---------------------------------|:---------|:-------------------|:--------------|:-------------------|:---------------|:----------------------|:-------------------|
| 71cb366f120340f199c9573c655c0901 | FINISHED |                    |               |                    |                |                       |                    |

### 08_mlflow_xgboost

| run_id                           | status   |   metrics.accuracy |   metrics.auc | metrics.f1_macro   | params.model   |   params.n_estimators |   params.max_depth |
|:---------------------------------|:---------|-------------------:|--------------:|:-------------------|:---------------|----------------------:|-------------------:|
| a80c536d9316462db684a836fc76a8ca | FINISHED |            0.95614 |      0.994709 |                    |                |                   300 |                  4 |
| 7ae57a0be5d84f4f917b48c7c373e080 | FINISHED |            0.95614 |      0.994709 |                    |                |                   300 |                  4 |
