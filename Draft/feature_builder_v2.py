import re
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import featuretools as ft


# application_train_df = pd.read_parquet("../dataset/sample_100k/app_train.parquet")
# application_test_df = pd.read_parquet("../dataset/sample_100k/app_test.parquet")


# bureau_df = pd.read_csv("../home-credit-default-risk/bureau.csv")
# previous_application_df = pd.read_csv("../home-credit-default-risk/previous_application.csv")

class DatasetType(Enum):
    BASE = 0
    FULL = 1

    BASE_ONLY_CREDIT_RATINGS = 100


class TargetDataset(Enum):
    TRAIN = 0
    TEST = 1


def aggregate_previous_applications():
    pos_cash_balance = pd.read_parquet("../dataset/full/POS_CASH_balance.parquet")
    previous_application_df = pd.read_parquet("../dataset/full/previous_application.parquet")
    # Calculate total DPD per loan
    total_dpd = pos_cash_balance.groupby('SK_ID_PREV')['SK_DPD_DEF'].sum().rename('total_dpd')

    # Identify loans with any DPD > 0 and calculate proportion
    pos_cash_balance['has_dpd'] = pos_cash_balance['SK_DPD_DEF'] > 0
    loans_with_any_dpd = pos_cash_balance.groupby('SK_ID_PREV')['has_dpd'].max().astype(int).rename('has_any_dpd')

    # Calculate the proportion of months with DPD > 0 for each loan
    months_with_dpd_prop = pos_cash_balance.groupby('SK_ID_PREV')['has_dpd'].mean().rename('months_with_dpd_prop')

    # Aggregate these metrics into a new DataFrame
    dpd_aggregations = pd.concat([total_dpd, loans_with_any_dpd, months_with_dpd_prop], axis=1).reset_index()

    # Merge aggregated DPD metrics with previous_application_df
    previous_application_df = previous_application_df.merge(dpd_aggregations, on='SK_ID_PREV', how='left')

    # Fill missing values for loans without POS CASH balance entries
    previous_application_df[['total_dpd', 'months_with_dpd_prop']] = previous_application_df[
        ['total_dpd', 'months_with_dpd_prop']].fillna(0)
    previous_application_df['has_any_dpd'] = previous_application_df['has_any_dpd'].fillna(0)

    # Basic Aggregations
    aggregations = {
        'AMT_APPLICATION': ['mean', 'sum'],
        'AMT_CREDIT': ['mean', 'sum'],
        'AMT_DOWN_PAYMENT': 'sum',
        'RATE_INTEREST_PRIMARY': ['mean', 'std'],
    }
    aggregations.update({
        'total_dpd': 'sum',
        'has_any_dpd': ['mean', 'sum'],  # Proportion of loans with any DPD and count of loans with DPD
        'months_with_dpd_prop': ['mean', 'sum']  # Average and total proportion of months with DPD across loans
    })

    # Aggregate data
    agg_df = previous_application_df.groupby('SK_ID_CURR').agg(aggregations)
    agg_df.columns = ['prev_' + '_'.join(col).strip() for col in agg_df.columns.values]

    # Calculate additional derived variables
    agg_df['prev_total_previous_loans'] = previous_application_df.groupby('SK_ID_CURR')['SK_ID_PREV'].transform('count')
    agg_df['prev_credit_received_requested_diff'] = agg_df['prev_AMT_CREDIT_sum'] - agg_df['prev_AMT_APPLICATION_sum']
    agg_df['prev_ratio_sum_down_payment_credit'] = agg_df['prev_AMT_DOWN_PAYMENT_sum'] / agg_df['prev_AMT_CREDIT_sum']

    # Handling of categorical and datetime variables
    # Filtering for last loans based on FLAG_LAST_APPL_PER_CONTRACT
    last_loans_df = previous_application_df.sort_values(by=['SK_ID_CURR', 'DAYS_DECISION']).drop_duplicates(
        'SK_ID_CURR', keep='last')

    # Adding last loan specific variables
    agg_df['prev_last_loan_interest_rate'] = last_loans_df.set_index('SK_ID_CURR')['RATE_INTEREST_PRIMARY']
    agg_df['prev_last_loan_purpose'] = last_loans_df.set_index('SK_ID_CURR')['NAME_CASH_LOAN_PURPOSE']
    agg_df['prev_last_loan_contract_status'] = last_loans_df.set_index('SK_ID_CURR')['NAME_CONTRACT_STATUS']
    agg_df['prev_last_loan_decision_date'] = last_loans_df.set_index('SK_ID_CURR')['DAYS_DECISION']
    agg_df['prev_last_loan_payment_type'] = last_loans_df.set_index('SK_ID_CURR')['NAME_PAYMENT_TYPE']
    agg_df['prev_last_loan_code_reject_reason'] = last_loans_df.set_index('SK_ID_CURR')['CODE_REJECT_REASON']
    agg_df['prev_last_loan_client_type'] = last_loans_df.set_index('SK_ID_CURR')['NAME_CLIENT_TYPE']
    agg_df['prev_last_loan_portfolio'] = last_loans_df.set_index('SK_ID_CURR')['NAME_PORTFOLIO']
    agg_df['prev_last_loan_goods_category'] = last_loans_df.set_index('SK_ID_CURR')['NAME_GOODS_CATEGORY']
    agg_df['prev_last_loan_product_type'] = last_loans_df.set_index('SK_ID_CURR')['NAME_PRODUCT_TYPE']
    agg_df['prev_last_loan_yield_group'] = last_loans_df.set_index('SK_ID_CURR')['NAME_YIELD_GROUP']

    # NAME_CONTRACT_STATUS count per category with prefix
    contract_status_counts = previous_application_df.pivot_table(index='SK_ID_CURR', columns='NAME_CONTRACT_STATUS',
                                                                 aggfunc='size', fill_value=0)
    # Add prefix to column names
    contract_status_counts.columns = ['prev_contract_status_' + str(col) + '_count' for col in
                                      contract_status_counts.columns]
    agg_df = agg_df.join(contract_status_counts, on='SK_ID_CURR')

    # NAME_PORTFOLIO counts with prefix
    portfolio_counts = previous_application_df.pivot_table(index='SK_ID_CURR', columns='NAME_PORTFOLIO', aggfunc='size',
                                                           fill_value=0)
    # Add prefix to column names
    portfolio_counts.columns = ['prev_portfolio_' + str(col) + '_count' for col in portfolio_counts.columns]
    agg_df = agg_df.join(portfolio_counts, on='SK_ID_CURR')

    # NAME_PRODUCT_TYPE counts with prefix
    product_type_counts = previous_application_df.pivot_table(index='SK_ID_CURR', columns='NAME_PRODUCT_TYPE',
                                                              aggfunc='size', fill_value=0)
    # Add prefix to column names
    product_type_counts.columns = ['prev_product_type_' + str(col) + '_count' for col in product_type_counts.columns]
    agg_df = agg_df.join(product_type_counts, on='SK_ID_CURR')

    # Filter out "XNA" values from the yield group before mapping and calculating the average
    yield_mapping = {
        "middle": 1,
        "high": 2,
        "low_normal": 0,
        "low_action": 0,
    }

    # Create a filtered DataFrame where 'NAME_YIELD_GROUP' is not 'XNA'
    filtered_df = previous_application_df[previous_application_df['NAME_YIELD_GROUP'] != 'XNA'].copy()

    # Map the yield groups to ordinal values using the filtered DataFrame
    filtered_df['yield_group_ordinal'] = filtered_df['NAME_YIELD_GROUP'].map(yield_mapping)

    # Calculate the average yield group for each client, excluding 'XNA' values
    agg_df['prev_avg_yield_group'] = filtered_df.groupby('SK_ID_CURR')['yield_group_ordinal'].mean()

    # Days after first loan application and currently active loans
    agg_df['prev_days_after_first_application'] = previous_application_df.groupby('SK_ID_CURR')[
        'DAYS_DECISION'].transform('min')
    agg_df['prev_currently_active_loans'] = \
        previous_application_df[previous_application_df['DAYS_TERMINATION'] > 0].groupby('SK_ID_CURR')[
            'DAYS_TERMINATION'].transform('count').fillna(0)

    # First, calculate counts for each NAME_CONTRACT_STATUS category for each client
    status_counts = previous_application_df.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].value_counts().unstack(
        fill_value=0)

    # Ensure all necessary columns are present, even if they don't exist in the data
    for status in ['Approved', 'Canceled', 'Refused', 'Unused offer']:
        if status not in status_counts:
            status_counts[status] = 0

    # Add prefixes and assign counts to agg_df
    prefixed_status_counts = status_counts.add_prefix('prev_').add_suffix('_loans')
    agg_df = agg_df.join(prefixed_status_counts, on='SK_ID_CURR')

    # Calculate total loans
    agg_df['prev_total_loans'] = agg_df[
        ['prev_Approved_loans', 'prev_Canceled_loans', 'prev_Refused_loans', 'prev_Unused offer_loans']].sum(axis=1)

    # Calculate ratios for each NAME_CONTRACT_STATUS category to total loans
    agg_df['prev_accepted_to_total_ratio'] = agg_df['prev_Approved_loans'] / agg_df['prev_total_loans']
    agg_df['prev_cancelled_to_total_ratio'] = agg_df['prev_Canceled_loans'] / agg_df['prev_total_loans']
    agg_df['prev_refused_to_total_ratio'] = agg_df['prev_Refused_loans'] / agg_df['prev_total_loans']
    agg_df['prev_unused_to_total_ratio'] = agg_df['prev_Unused offer_loans'] / agg_df['prev_total_loans']

    # Recalculate the ratio of rejected (here, considered as Refused) to accepted (Approved) loans
    # Note: Adding 1 to the denominator to ensure we don't divide by zero
    # agg_df['prev_ratio_rejected_accepted'] = agg_df['prev_Refused_loans'] / (agg_df['prev_Approved_loans'] + 1)
    conditions = [
        (agg_df['prev_Refused_loans'] == 0) & (agg_df['prev_Approved_loans'] == 0),  # No applications
        (agg_df['prev_Approved_loans'] > 0) & (
                    agg_df['prev_Refused_loans'].isnull() | (agg_df['prev_Refused_loans'] == 0)),  # Approved only
        (agg_df['prev_Approved_loans'] == 0) & (agg_df['prev_Refused_loans'] > 0)  # Refused only
    ]
    choices = [
        np.nan,  # Output NaN for no applications
        0,  # Output 0 for approved only
        1  # Output 1 for refused only
    ]

    agg_df['prev_ratio_rejected_accepted'] = np.select(conditions, choices, default=agg_df['prev_Refused_loans'] / (
                agg_df['prev_Approved_loans'] + agg_df['prev_Refused_loans']))

    # Renaming columns where necessary to match provided list
    # Note: Some variables such as counts per NAME_CONTRACT_STATUS and NAME_PORTFOLIO have been added directly to the agg_df DataFrame
    # and their names adjusted according to your instructions, with '_count' suffixes to denote the specific counts.

    # Ensure all new column names are correctly prefixed and match your requirements
    # This block of code sets a strong foundation. You might need to adjust column names or calculations to fit your exact specifications or handle edge cases.

    agg_df['prev_last_loan_code_reject_reason'] = last_loans_df.set_index('SK_ID_CURR')['CODE_REJECT_REASON']
    # Count loans by CODE_REJECT_REASON for each client
    reject_reason_counts = previous_application_df.pivot_table(index='SK_ID_CURR', columns='CODE_REJECT_REASON',
                                                               aggfunc='size', fill_value=0)

    # Add prefix to column names to reflect their origin
    reject_reason_counts.columns = ['prev_code_reject_reason_' + str(col) + '_count' for col in
                                    reject_reason_counts.columns]

    # Merge these counts into agg_df
    agg_df = agg_df.join(reject_reason_counts, on='SK_ID_CURR')

    # Step 1: Add the last value of NFLAG_INSURED_ON_APPROVAL for each client's most recent loan
    agg_df['prev_last_loan_nflag_insured_on_approval'] = last_loans_df.set_index('SK_ID_CURR')[
        'NFLAG_INSURED_ON_APPROVAL']

    # Step 2: Calculate the average value of NFLAG_INSURED_ON_APPROVAL for all loans for each client
    # Convert NFLAG_INSURED_ON_APPROVAL to numeric (0 or 1) if not already and calculate the average
    agg_df['prev_avg_nflag_insured_on_approval'] = previous_application_df.groupby('SK_ID_CURR')[
        'NFLAG_INSURED_ON_APPROVAL'].mean()

    return agg_df


TARGET_FILE = "full"  # sample_100k

def load_datasets_and_prepare_features(drop_meta_data=False,
                                       ds_type=DatasetType.BASE,
                                       ds_source=TargetDataset.TRAIN,
                                       drop_cols: List[str] = None,
                                       drop_cols_post_proc: List[str] = None
                                       ) -> pd.DataFrame:
    """
    Loads datasets and prepares features based on selected source dfs and features sets.
    :return:
    """
    if ds_source.value == TargetDataset.TRAIN.value:
        application_df = pd.read_parquet(f"../dataset/{TARGET_FILE}/app_train.parquet")
    elif ds_source.value == TargetDataset.TEST.value:
        application_df = pd.read_parquet(f"../dataset/{TARGET_FILE}/app_test.parquet")
    else:
        raise NotImplementedError()

    # application_df = application_df.sample(1000, random_state=42)

    if ds_type.value == DatasetType.FULL.value:
        bureau_df = pd.read_parquet("../dataset/full/bureau.parquet")

        # raise Exception("TODO")
        # Initialize an EntitySet
        application_df = application_df.copy()
        es = ft.EntitySet(id="Clients")

        # Add the applications dataframe to the entity set
        es = es.add_dataframe(dataframe_name="applications",
                              dataframe=application_df,
                              index="SK_ID_CURR",
                              logical_types={"SK_ID_CURR": "Integer"})

        # Add the bureau dataframe to the entity set
        es = es.add_dataframe(dataframe_name="bureau",
                              dataframe=bureau_df,
                              index="SK_BUREAU_ID",
                              logical_types={"SK_ID_CURR": "Integer"})

        # Define and add the relationship
        es = es.add_relationship("applications", "SK_ID_CURR", "bureau", "SK_ID_CURR")

        # Define custom primitives if necessary or use built-in primitives for direct feature generation
        # For this scenario, we'll leverage built-in primitives and then perform additional pandas operations for specific needs

        # Generate features with DFS

        # Generate features with DFS, including more primitives as needed
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="applications",
            agg_primitives=["count", "mean", "max", "min"],  # Added more agg_primitives for diversity
            trans_primitives=[],
            drop_contains=["SK_ID_PREV"],  # Example of dropping irrelevant features
            max_depth=2
        )

        # Since we should not modify the original bureau_df, perform calculations separately and merge them into the feature_matrix

        # Calculating additional features outside of FeatureTools processing

        # Active loans count per customer
        active_loans_count = bureau_df[bureau_df["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").size().reset_index(
            name="active_loans_count")

        # Total defaults and default ratio
        defaults = bureau_df[bureau_df["CREDIT_DAY_OVERDUE"] > 0].groupby("SK_ID_CURR").size().reset_index(
            name="total_defaults")
        defaults["default_ratio"] = defaults["total_defaults"] / \
                                    bureau_df.groupby("SK_ID_CURR").size().reset_index(name="total_loans")[
                                        "total_loans"]

        # Last loan issued based on DAYS_ENDDATE_FACT
        last_loan_issued = bureau_df.groupby("SK_ID_CURR")["DAYS_ENDDATE_FACT"].max().reset_index(
            name="last_loan_issued_days")

        # Merge these calculations into the feature_matrix
        feature_matrix = feature_matrix.reset_index().merge(active_loans_count, on="SK_ID_CURR", how="left")
        feature_matrix = feature_matrix.merge(defaults[["SK_ID_CURR", "total_defaults", "default_ratio"]],
                                              on="SK_ID_CURR",
                                              how="left")

        feature_matrix = feature_matrix.merge(last_loan_issued, on="SK_ID_CURR", how="left")

        feature_matrix['last_loan_issued_days'] = feature_matrix['last_loan_issued_days'].fillna(
            feature_matrix['last_loan_issued_days'].max() * 5)

        # Fill NaN values for cases with no active loans or defaults
        feature_matrix.fillna(
            {"active_loans_count": 0,
             "total_defaults": 0,
             "default_ratio": 0},
            inplace=True)

        print("Appending previous history")
        agg_df = aggregate_previous_applications()
        feature_matrix = feature_matrix.merge(agg_df, on='SK_ID_CURR', how='left')

    elif ds_type.value == DatasetType.BASE_ONLY_CREDIT_RATINGS.value:
        feature_matrix = application_df[["TARGET", "SK_ID_CURR", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].copy()
    else:
        feature_matrix = application_df.copy()

    if drop_cols is not None:
        feature_matrix = feature_matrix.drop(columns=drop_cols)

    feature_matrix = feature_matrix.rename(columns={"TARGET": "TARGET"})

    if drop_meta_data:
        feature_matrix = feature_matrix.drop(columns=["SK_ID_CURR"])

    for col in feature_matrix.select_dtypes(include='object').columns:
        feature_matrix[col] = feature_matrix[col].astype('category')

    # Display the first few rows of the feature matrix to verify the result

    def clean_and_camel_case(s):
        if s == "TARGET":
            return s

        # Remove non-alphanumeric characters except underscore
        s = re.sub(r'[^a-zA-Z0-9_]', '', s)  # Remove non-alphanumeric characters
        parts = s.split('_')
        return ''.join(part.capitalize() for part in parts)

    column_values = feature_matrix.columns

    column_map = {}
    for i, c in enumerate(column_values):
        column_map[c] = str(i)
    # Apply the function to each column name
    feature_matrix.columns = [clean_and_camel_case(col) for col in feature_matrix.columns]
    # test_df.columns = [clean_and_camel_case(col) for col in test_df.columns]

    if drop_cols_post_proc is not None:
        print(f"drop drop_cols_post_proc: {len(feature_matrix.columns)}")
        feature_matrix = feature_matrix.drop(
            columns=[col for col in drop_cols_post_proc if col in feature_matrix.columns], errors='ignore')
        print(f"after drop_cols_post_proc: {len(feature_matrix.columns)}")


    print(f"Full DS size: {len(feature_matrix)}")
    return feature_matrix
