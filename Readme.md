Notebooks and Model

[Notebboks: Table of Contents](https://qwyt.github.io/turing_m3_s3_publish/index.html](https://qwyt.github.io/ML.4.1_presentation/)


[Model and Analysis](https://qwyt.github.io/ML.4.1_presentation/2.0_Model.html)


### Feature Engineering

#### Feature Selection and Aggregation

Feature Tools DFS is used for feature engineering in combination with manual aggregations. Currently, these include:

- bureau.csv
- previous_application.csv
- credit_card_balance.csv

#### Evaluation and Baseline

The upper range baseline (i.e. the best performing submission on Kaggle) is slightly over AUC = 0.8, we can't reasonably
expect to beat it so our goal is to get as close as possible.

The lower range baseline is a simple classification model that's only using credit rating/score columns
EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3. It achieves an AUC of ~ 0.72. This mean that the range between them is
ralatively narrow which means that even a seemingly small increase in AUC e.g. by 0.01 would be pretty signficant.

`_*An AUC of 0.5 suggests a model that performs no better than random guessing*_`

### Goals And Processes

In addition to maximizing the AUC score we have attempted to use the model for more practical purpose of estimating loan
quality grades (e.g. A, B, C, D etc.) based on default risk likelihood and general banking practices.


#### Explainability

We've used LGBM which is relatively complex "blackbox" model which might not be the ideal in loan evaluations and similar tasks because it's hard to objectively explain the specific decisions the model made (based on regulatory or customer related requirements). 

However, we believe that we were largely able to overcome this shortcoming through the use of single observation SHAP plots:

they allow us to attribute the impact of specific feature (e.g. credit scores, client income etc.) on the estimated risk which allows to select an appropriate grade, interest rate and decide whether the loan should or should not be approved based on our acceptable risk preferences: 


### Pipeline and Technical Details


#### Model selection
- We have started with a wider group of models such as Logit, XGBoost, CatBoost and LGBM we have found that LGBM provided the best performance and training speed out of the box and some initial tuning so it was selection for our production model (we've tried different approaches like combining the outcomes of XGBoost and LGBM into an ensemble model but this has provided poorer performance)

- Feature engineering and selection:
  - All the features from the `application_train.csv` are included
  - Specific grouped and/or aggregated features are created  manually (based on subject knowledge) from the `bureau.csv` and `previous_applications.csv` (e.g. based on main client, default loans, rejected applications, repayment history etc.)
  - `Featuretools Deep Feature Synthesis` package is used to generate a large number of aggregated features which might be potentially useful for the analysis.
 - General clean up and processing is performed (however almost no data imputation was done for missing values because in most cases they seem to represent valid data points (e.g. `car age` = `NaN` etc.) and because complex models like LGBM handle his internally we judged it to be not necessary).

- Initial Baseline Models:
 - We have started with two LGBM models `Base` (only 121 features included features from the `application_test.csv) and `Full` (224 features, based on manual and Featuretools aggregations).


#### Tuning:
- Bayesian optimization with Optuna
- Multi objective optimization (i.e. for auc, pr-auc, f1, time) [TODO]
