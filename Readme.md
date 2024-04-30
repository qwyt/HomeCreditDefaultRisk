Notebooks and Model

[Notebooks: Table of Contents](https://qwyt.github.io/ML.4.1_presentation/)


[Model and Analysis](https://qwyt.github.io/ML.4.1_presentation/2.0_Model.html)

## Summary

| Model                            |   auc | pr_auc | target_f1 | target_recall | target_precision |
|----------------------------------|-------|--------|-----------|---------------|------------------|
| LGBM (dart)                      | 0.775 |  0.263 |     0.291 |         0.654 |            0.187 |
| LGBM (                           | 0.774 |  0.261 |     0.290 |         0.661 |            0.186 |
| LGBM (all features)              | 0.773 |  0.262 |     0.290 |         0.658 |            0.186 |
| LGBM (only applications ds)      | 0.759 |  0.243 |     0.273 |         0.661 |            0.172 |
| Baseline_Only_CreditRatings      | 0.723 |  0.202 |     0.244 |         0.657 |            0.150 |
| Best Kaggle Competition Models   | 0.8   |   --   |      --   |          --   |             --   |


### Feature Engineering

#### Feature Selection and Aggregation

Feature Tools DFS was  used for feature engineering in combination with manual aggregations. Currently, these include:

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

### Additional Goals

In addition to building a classification model we've used the estimating propabilities to:
- Build a Risk-Based Pricing model (using calibrarted thresholds) for determining optimal interest rates (relative to an arbitrary expected portfolio rate of return).
- Assign loan quality grades based on default risk.
- Calculate overal and per grade hypothetical portfolio returns. 
- Calculate change in portfolio returns if the selected LGBM model is used to accepted/reject loan applications.

|                         | Actual       | Hypothetical |
|-------------------------|--------------|--------------|
| Total Loan Amount       | 18503.7355M  | 13522.2582M  |
| Total Interest Paid     | 17.3487M     | 13.7148M     |
| Total Return %          | 0.09%        | 0.10%        |
| Default Rate            | 8.30%        | 2.96%        |
| Total Loss              | 1435.5231M   | 208.7593M    |
| Losses Avoided          | None         | 1226.7638M   |
| Interest Lost           | None         | -3.6339M     |
| Total Applications Accepted | 30752    | 21442        |


Using our model to employ a more conservative lending strategy would pottentially allow Home Credit to decrease their loses by up to ~80% (based on hypotehetical base interest rate, calculation would need to be performed using actual interest rates offered by Home Credit and LGD ratio).

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
- Bayesian optimization with Optuna (around 100-150 trials depending on specific model)
