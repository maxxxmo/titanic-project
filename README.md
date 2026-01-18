# Project overview




## Phase 1 - Core ML Pipeline (Local Execution)


### Design choice – No inheritance

We avoid inheritance for models and metrics. Since we rely on scikit-learn components that we do not own, using inheritance would add unnecessary abstraction and duplicate existing APIs. Instead, the pipeline is configuration-driven (model and metric registries) with generic training and evaluation logic.


# Phase 2 - Add mlflow

- Titanic ML pipeline tracked with MLflow: parameters, metrics, and model.

- Uses a metrics registry to centralize metrics (accuracy, precision, recall, f1, roc_auc).

- Each training run is logged in an MLflow Experiment, no Model Registry used.

- Pipeline can be run with python src/run_pipeline.py, fully reproducible.


# Phase 3 Continuous Integration

***Linter:***  Tools used to automatically analyze code and flag style or potential error issues

What i want is every push to verify the code, flake8 for style, pylint to checks for errors and score code and black for formating and indentation.

I also add in /tests/units Unitary test for differents steps of the pipeline to check outputs sizes and formats. NaN presence. These test will be done with pytest.


# Phase 4 Airflow

## What is a DAG

- DAG = Directed Acyclic Graph = Graphe Orienté Acyclique
--> Soit un graphe avec des flèches directionnelles sans boucles
--> C'est une représentation du pipeline comme un graphe de tâche






Architecture : 






Then we can test it by running docker desktop or docker compose (but im on windows so i will prefer docker desktop)



# Phase 5 Continuous Integration