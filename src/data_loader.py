import pandas as pd
import numpy as np
import gc
import os


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics to reduce memory usage."""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def load_main_tables(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load application_train and application_test.
    Returns (app_train, app_test)
    """
    print("Loading main application tables...")

    app_train = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'application_train.csv'))
    )
    app_test = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'application_test.csv'))
    )

    print(f"  app_train: {app_train.shape[0]:,} rows  "
          f"{app_train.shape[1]} cols  "
          f"{app_train.memory_usage().sum()/1e6:.1f} MB")
    print(f"  app_test:  {app_test.shape[0]:,} rows  "
          f"{app_test.shape[1]} cols  "
          f"{app_test.memory_usage().sum()/1e6:.1f} MB")

    gc.collect()
    return app_train, app_test


def load_bureau(data_dir: str) -> pd.DataFrame:
    """Load bureau.csv with memory optimization."""
    print("Loading bureau...")
    df = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'bureau.csv'))
    )
    print(f"  bureau: {df.shape[0]:,} rows  {df.shape[1]} cols  "
          f"{df.memory_usage().sum()/1e6:.1f} MB")
    return df


def load_bureau_balance(data_dir: str) -> pd.DataFrame:
    """Load bureau_balance.csv with memory optimization."""
    print("Loading bureau_balance...")
    df = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'bureau_balance.csv'))
    )
    print(f"  bureau_balance: {df.shape[0]:,} rows  {df.shape[1]} cols  "
          f"{df.memory_usage().sum()/1e6:.1f} MB")
    return df


def load_previous_application(data_dir: str) -> pd.DataFrame:
    """Load previous_application.csv with memory optimization."""
    print("Loading previous_application...")
    df = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'previous_application.csv'))
    )
    print(f"  previous_application: {df.shape[0]:,} rows  {df.shape[1]} cols  "
          f"{df.memory_usage().sum()/1e6:.1f} MB")
    return df


def load_installments(data_dir: str) -> pd.DataFrame:
    """Load installments_payments.csv with memory optimization."""
    print("Loading installments_payments...")
    df = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'installments_payments.csv'))
    )
    print(f"  installments: {df.shape[0]:,} rows  {df.shape[1]} cols  "
          f"{df.memory_usage().sum()/1e6:.1f} MB")
    return df


def load_credit_card(data_dir: str) -> pd.DataFrame:
    """Load credit_card_balance.csv with memory optimization."""
    print("Loading credit_card_balance...")
    df = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'credit_card_balance.csv'))
    )
    print(f"  credit_card: {df.shape[0]:,} rows  {df.shape[1]} cols  "
          f"{df.memory_usage().sum()/1e6:.1f} MB")
    return df


def load_pos_cash(data_dir: str) -> pd.DataFrame:
    """Load POS_CASH_balance.csv with memory optimization."""
    print("Loading POS_CASH_balance...")
    df = reduce_memory(
        pd.read_csv(os.path.join(data_dir, 'POS_CASH_balance.csv'))
    )
    print(f"  pos_cash: {df.shape[0]:,} rows  {df.shape[1]} cols  "
          f"{df.memory_usage().sum()/1e6:.1f} MB")
    return df


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    app_train, app_test = load_main_tables(DATA_DIR)
    print("\ndata_loader.py working correctly")