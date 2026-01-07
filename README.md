# Project overview




## Phase 1 - Core ML Pipeline (Local Execution)


### Design choice – No inheritance

We avoid inheritance for models and metrics. Since we rely on scikit-learn components that we do not own, using inheritance would add unnecessary abstraction and duplicate existing APIs. Instead, the pipeline is configuration-driven (model and metric registries) with generic training and evaluation logic.


# Phase 2 - Add mlflow
