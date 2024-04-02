



### Feature Engineering


#### Feature Selection and Aggregation

Feature Tools DFS is used for feature engineering in combination with manual aggregations. Currently, these include:

- bureau.csv
- previous_application.csv
- credit_card_balance.csv


#### Evaluation and Baseline 

The upper range baseline (i.e. the best performing submission on Kaggle) is slightly over AUC = 0.8, we can't reasonable expect to beat it so our goal is to get as close as possible.

Additionally, we've included a model selected using EvalML (and auto ML library) and a raw dataset (with Featuretools aggregations etc.)

#### Clustering

