### Feature Engineering

#### Feature Selection and Aggregation

Feature Tools DFS is used for feature engineering in combination with manual aggregations. Currently, these include:

- bureau.csv
- previous_application.csv
- credit_card_balance.csv
-

#### Evaluation and Baseline

The upper range baseline (i.e. the best performing submission on Kaggle) is slightly over AUC = 0.8, we can't reasonably
expect to beat it so our goal is to get as close as possible.

The lower range baseline is a simple classification model that's only using credit rating/score columns
EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3. It achieves an AUC of ~ 0.72. This mean that the range between them is ralatively narrow which means that even a seemingly small increase in AUC e.g. by 0.01 would be pretty signficant. 

`_*An AUC of 0.5 suggests a model that performs no better than random guessing*_`

Additionally, we've included a model selected using EvalML (and auto ML library) and a raw dataset (with Featuretools
aggregations etc.)

#### Clustering

Clustering before using XGBoost can simplify data and possibly improve model performance by highlighting patterns that
XGBoost may overlook. This preprocessing step reduces dimensionality and can enhance model interpretability, but its
effectiveness depends on data relevance and feature importance evaluation.

##### K-Prototypes

Most suitable the dataset has clear boundaries and a roughly uniform distribution for optimal results. We've been unable
to obtain clearly defined cluster when using it and based on the type of the dataset it's probably not the most suitable
algorithm.

#### DBSCAN

Is an unsupervised algorithm which is more suitable for datasets with significant noise or irrelevant data points (e.g.
data exhibits non-globular or irregularly shaped clusters)

Feature

- We've experimented

#https://www.kaggle.com/c/home-credit-default-risk/data

#### Tuning:
- Bayesian optimization with Optuna
- Multi objective optimization (i.e. for auc, pr-auc, f1, time) [TODO]