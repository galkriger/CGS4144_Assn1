# Assignment 4 logistic regression

L1 logistic regression (C=1, balanced class weights) predicts AL/RF and the three hierarchical clusters from selected_clusters.csv.

Five-fold cross-validation keeps runs with the same title together. Gene selection and scaling use training data only.

| Genes | AL/RF AUC | Cluster AUC (macro OvR) |
|---:|---:|---:|
| 10 | 0.9591 | 0.8133 |
| 100 | 0.9813 | 0.9668 |
| 1000 | 0.9984 | 0.9875 |
| 5000 | 0.9995 | 0.9956 |
| 10000 | 0.9992 | 0.9952 |

Scores use held-out predictions. Cluster AUC measures how well the model recovers the Assignment 3 labels.

Both tasks improve up to 5,000 genes; 10,000 genes gives no further improvement.

Plots and short explanations are in [assignment4_logistic.ipynb](../../../assignment4_logistic.ipynb). Detailed scores and predictions are saved in metrics.csv and predictions.csv.

To rerun, open the notebook from the project folder and choose Run All.
