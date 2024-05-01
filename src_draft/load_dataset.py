import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def _internal_split_and_save_dfs(
    df: pd.DataFrame, base_output_dir: str, ds_key: str, no_split=False
):
    def save_parquet(df, subdir, filename):
        output_dir = os.path.join(base_output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_parquet(filepath)
        print(f"Rows in {filepath}: {len(df)}")

    def sample_split_save(df, sample_size, test_size, subdir):
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        # df_tuning, df_train_test = train_test_split(
        #     df_sample, test_size=test_size, random_state=42
        # )

        # save_parquet(df_tuning, subdir, f"{ds_key}_tuning.parquet")
        save_parquet(df_sample, subdir, f"{ds_key}.parquet")

    # df_tuning, df_train_test = train_test_split(df, test_size=0.9, random_state=42)
    save_parquet(df, "full", f"{ds_key}.parquet")

    if not no_split:
        sample_split_save(df, 100000, 0.9, "sample_100k")
        print(len(df))
        sample_split_save(df, 20000, 0.9, "sample_20k")


def split_datasets_combined_accepted_rejected(
    base_output_dir="../dataset",
):
    FILE_PATH_TRAIN = "../home-credit-default-risk/application_train.csv"
    FILE_PATH_TEST = "../home-credit-default-risk/application_test.csv"

    app_train_df = pd.read_csv(FILE_PATH_TRAIN)
    app_test_df = pd.read_csv(FILE_PATH_TEST)

    print(f"Total Lines in Source application_train: {len(app_train_df)}")
    print(f"Total Lines in Source application_test: {len(app_test_df)}")

    _internal_split_and_save_dfs(
        app_train_df, base_output_dir=base_output_dir, ds_key="app_train"
    )
    _internal_split_and_save_dfs(
        app_test_df, base_output_dir=base_output_dir, ds_key="app_test"
    )

    FILE_PATH_BUREU = "../home-credit-default-risk/bureau.csv"
    bureau_df = pd.read_csv(FILE_PATH_BUREU)
    _internal_split_and_save_dfs(
        bureau_df, base_output_dir=base_output_dir, ds_key="bureau", no_split=True
    )

    FILE_PATH_BUREU = "../home-credit-default-risk/bureau_balance.csv"
    bureau_df = pd.read_csv(FILE_PATH_BUREU)
    _internal_split_and_save_dfs(
        bureau_df,
        base_output_dir=base_output_dir,
        ds_key="bureau_balance",
        no_split=True,
    )

    FILE_PATH_BUREU = "../home-credit-default-risk/credit_card_balance.csv"
    bureau_df = pd.read_csv(FILE_PATH_BUREU)
    _internal_split_and_save_dfs(
        bureau_df,
        base_output_dir=base_output_dir,
        ds_key="credit_card_balance",
        no_split=True,
    )

    FILE_PATH_BUREU = "../home-credit-default-risk/installments_payments.csv"
    bureau_df = pd.read_csv(FILE_PATH_BUREU)
    _internal_split_and_save_dfs(
        bureau_df,
        base_output_dir=base_output_dir,
        ds_key="installments_payments",
        no_split=True,
    )

    FILE_PATH_BUREU = "../home-credit-default-risk/previous_application.csv"
    bureau_df = pd.read_csv(FILE_PATH_BUREU)
    _internal_split_and_save_dfs(
        bureau_df,
        base_output_dir=base_output_dir,
        ds_key="previous_application",
        no_split=True,
    )

    FILE_PATH_BUREU = "../home-credit-default-risk/POS_CASH_balance.csv"
    bureau_df = pd.read_csv(FILE_PATH_BUREU)
    _internal_split_and_save_dfs(
        bureau_df,
        base_output_dir=base_output_dir,
        ds_key="POS_CASH_balance",
        no_split=True,
    )


if __name__ == "__main__":
    split_datasets_combined_accepted_rejected()
