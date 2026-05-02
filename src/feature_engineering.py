import pandas as pd
import numpy as np
import gc
from src.data_loader import (
    load_bureau,
    load_bureau_balance,
    load_previous_application,
    load_installments,
    load_credit_card,
    load_pos_cash
)


def process_main_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from main application table.
    - Fix DAYS_EMPLOYED anomaly
    - Create ratio features
    - Create missingness flag columns
    """
    print("Processing main application table...")
    df = df.copy()

    # Fix DAYS_EMPLOYED anomaly — replace 365243 with NaN and flag it
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # Convert negative days to positive years — more interpretable
    df['AGE_YEARS']              = -df['DAYS_BIRTH']      / 365
    df['EMPLOYED_YEARS']         = -df['DAYS_EMPLOYED']   / 365
    df['REGISTRATION_YEARS']     = -df['DAYS_REGISTRATION'] / 365
    df['ID_PUBLISH_YEARS']       = -df['DAYS_ID_PUBLISH']  / 365

    # Credit ratio features — key risk indicators
    df['CREDIT_INCOME_RATIO']    = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO']   = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM']            = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['GOODS_CREDIT_RATIO']     = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)

    # EXT_SOURCE combined features
    df['EXT_SOURCE_MEAN']        = df[['EXT_SOURCE_1',
                                       'EXT_SOURCE_2',
                                       'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_MIN']         = df[['EXT_SOURCE_1',
                                       'EXT_SOURCE_2',
                                       'EXT_SOURCE_3']].min(axis=1)
    df['EXT_SOURCE_STD']         = df[['EXT_SOURCE_1',
                                       'EXT_SOURCE_2',
                                       'EXT_SOURCE_3']].std(axis=1)

    # Income per family member
    df['INCOME_PER_PERSON']      = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

    # Missingness indicator flags for high-missing columns
    high_missing = [
        'EXT_SOURCE_1', 'EXT_SOURCE_3', 'AMT_GOODS_PRICE',
        'AMT_ANNUITY', 'CNT_FAM_MEMBERS', 'OWN_CAR_AGE'
    ]
    for col in high_missing:
        df[f'{col}_MISSING'] = df[col].isnull().astype(int)

    print(f"  New features added. Shape: {df.shape}")
    return df


def aggregate_bureau(data_dir: str) -> pd.DataFrame:
    """
    Aggregate bureau and bureau_balance tables per SK_ID_CURR.
    Returns one row per customer with aggregated credit history features.
    """
    print("Aggregating bureau tables...")

    bureau = load_bureau(data_dir)
    bureau_bal = load_bureau_balance(data_dir)

    # Aggregate bureau_balance per SK_ID_BUREAU first
    bureau_bal_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(
        BUREAU_BAL_MONTHS_COUNT = ('MONTHS_BALANCE', 'count'),
        BUREAU_BAL_DPD_MEAN     = ('STATUS', lambda x: (x == 'C').mean()),
    ).reset_index()

    del bureau_bal
    gc.collect()

    # Merge bureau_balance aggregates into bureau
    bureau['SK_ID_BUREAU'] = bureau['SK_ID_BUREAU'].astype('int64')
    bureau_bal_agg['SK_ID_BUREAU'] = bureau_bal_agg['SK_ID_BUREAU'].astype('int64')
    bureau = bureau.merge(bureau_bal_agg, on='SK_ID_BUREAU', how='left')
    del bureau_bal_agg
    gc.collect()

    # Aggregate bureau per SK_ID_CURR
    agg_funcs = {
        'DAYS_CREDIT':              ['mean', 'min', 'max'],
        'CREDIT_DAY_OVERDUE':       ['mean', 'max'],
        'DAYS_CREDIT_ENDDATE':      ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE':   ['mean', 'max'],
        'CNT_CREDIT_PROLONG':       ['sum'],
        'AMT_CREDIT_SUM':           ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT':      ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE':   ['mean'],
        'DAYS_CREDIT_UPDATE':       ['mean'],
        'BUREAU_BAL_MONTHS_COUNT':  ['mean', 'sum'],
    }

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_funcs)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper()
                          for col in bureau_agg.columns]
    bureau_agg['BUREAU_LOAN_COUNT']   = bureau.groupby('SK_ID_CURR').size()
    bureau_agg['BUREAU_ACTIVE_COUNT'] = bureau[
        bureau['CREDIT_ACTIVE'] == 'Active'
    ].groupby('SK_ID_CURR').size()

    bureau_agg = bureau_agg.reset_index()

    del bureau
    gc.collect()

    print(f"  bureau_agg shape: {bureau_agg.shape}")
    return bureau_agg


def aggregate_previous_applications(data_dir: str) -> pd.DataFrame:
    """
    Aggregate previous_application table per SK_ID_CURR.
    Returns one row per customer.
    """
    print("Aggregating previous applications...")

    prev = load_previous_application(data_dir)

    # Replace anomalous values
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Credit utilization in previous loans
    prev['PREV_CREDIT_RATIO'] = prev['AMT_CREDIT'] / (prev['AMT_APPLICATION'] + 1)

    agg_funcs = {
        'AMT_ANNUITY':        ['mean', 'max'],
        'AMT_APPLICATION':    ['mean', 'max'],
        'AMT_CREDIT':         ['mean', 'sum'],
        'PREV_CREDIT_RATIO':  ['mean', 'min'],
        'DAYS_DECISION':      ['mean', 'min'],
        'CNT_PAYMENT':        ['mean', 'sum'],
    }

    prev_agg = prev.groupby('SK_ID_CURR').agg(agg_funcs)
    prev_agg.columns = ['PREV_' + '_'.join(col).upper()
                        for col in prev_agg.columns]

    prev_agg['PREV_APP_COUNT']      = prev.groupby('SK_ID_CURR').size()
    prev_agg['PREV_APPROVED_COUNT'] = prev[
        prev['NAME_CONTRACT_STATUS'] == 'Approved'
    ].groupby('SK_ID_CURR').size()
    prev_agg['PREV_REFUSED_COUNT']  = prev[
        prev['NAME_CONTRACT_STATUS'] == 'Refused'
    ].groupby('SK_ID_CURR').size()

    prev_agg = prev_agg.reset_index()

    del prev
    gc.collect()

    print(f"  prev_agg shape: {prev_agg.shape}")
    return prev_agg


def aggregate_installments(data_dir: str) -> pd.DataFrame:
    """
    Aggregate installments_payments per SK_ID_CURR.
    Captures payment behavior — late payments, underpayments.
    """
    print("Aggregating installments...")

    inst = load_installments(data_dir)

    # Payment difference — positive means paid more than due
    inst['PAYMENT_DIFF']  = inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']
    # Days late — positive means paid late
    inst['DAYS_LATE']     = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['DAYS_LATE']     = inst['DAYS_LATE'].clip(lower=0)

    agg_funcs = {
        'PAYMENT_DIFF':   ['mean', 'min', 'sum'],
        'DAYS_LATE':      ['mean', 'max', 'sum'],
        'AMT_INSTALMENT': ['mean', 'sum'],
        'AMT_PAYMENT':    ['mean', 'sum'],
        'NUM_INSTALMENT_VERSION': ['nunique'],
    }

    inst_agg = inst.groupby('SK_ID_CURR').agg(agg_funcs)
    inst_agg.columns = ['INST_' + '_'.join(col).upper()
                        for col in inst_agg.columns]
    inst_agg['INST_COUNT'] = inst.groupby('SK_ID_CURR').size()

    inst_agg = inst_agg.reset_index()

    del inst
    gc.collect()

    print(f"  inst_agg shape: {inst_agg.shape}")
    return inst_agg


def aggregate_credit_card(data_dir: str) -> pd.DataFrame:
    """
    Aggregate credit_card_balance per SK_ID_CURR.
    Captures revolving credit utilization behavior.
    """
    print("Aggregating credit card balance...")

    cc = load_credit_card(data_dir)

    cc['CC_UTILIZATION'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)

    agg_funcs = {
        'AMT_BALANCE':              ['mean', 'max'],
        'AMT_CREDIT_LIMIT_ACTUAL':  ['mean', 'max'],
        'CC_UTILIZATION':           ['mean', 'max'],
        'AMT_DRAWINGS_ATM_CURRENT':       ['mean', 'sum'],
        'SK_DPD':                   ['mean', 'max'],
        'SK_DPD_DEF':               ['mean', 'max'],
    }

    cc_agg = cc.groupby('SK_ID_CURR').agg(agg_funcs)
    cc_agg.columns = ['CC_' + '_'.join(col).upper()
                      for col in cc_agg.columns]
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    cc_agg = cc_agg.reset_index()

    del cc
    gc.collect()

    print(f"  cc_agg shape: {cc_agg.shape}")
    return cc_agg


def aggregate_pos_cash(data_dir: str) -> pd.DataFrame:
    """
    Aggregate POS_CASH_balance per SK_ID_CURR.
    Captures point-of-sale and cash loan behavior.
    """
    print("Aggregating POS CASH balance...")

    pos = load_pos_cash(data_dir)

    agg_funcs = {
        'MONTHS_BALANCE':   ['mean', 'min'],
        'SK_DPD':           ['mean', 'max'],
        'SK_DPD_DEF':       ['mean', 'max'],
        'CNT_INSTALMENT':   ['mean', 'sum'],
    }

    pos_agg = pos.groupby('SK_ID_CURR').agg(agg_funcs)
    pos_agg.columns = ['POS_' + '_'.join(col).upper()
                       for col in pos_agg.columns]
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    pos_agg = pos_agg.reset_index()

    del pos
    gc.collect()

    print(f"  pos_agg shape: {pos_agg.shape}")
    return pos_agg


def build_features(app_train: pd.DataFrame,
                   app_test: pd.DataFrame,
                   data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master function — runs all feature engineering steps and
    merges all aggregated tables into final train and test sets.
    """
    print("\n=== BUILDING FEATURE SET ===\n")

    # Step 1 — process main application table
    train = process_main_application(app_train)
    test  = process_main_application(app_test)

    # Step 2 — aggregate and merge supplementary tables
    bureau_agg = aggregate_bureau(data_dir)
    train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    test  = test.merge(bureau_agg,  on='SK_ID_CURR', how='left')
    del bureau_agg
    gc.collect()

    prev_agg = aggregate_previous_applications(data_dir)
    train = train.merge(prev_agg, on='SK_ID_CURR', how='left')
    test  = test.merge(prev_agg,  on='SK_ID_CURR', how='left')
    del prev_agg
    gc.collect()

    inst_agg = aggregate_installments(data_dir)
    train = train.merge(inst_agg, on='SK_ID_CURR', how='left')
    test  = test.merge(inst_agg,  on='SK_ID_CURR', how='left')
    del inst_agg
    gc.collect()

    cc_agg = aggregate_credit_card(data_dir)
    train = train.merge(cc_agg, on='SK_ID_CURR', how='left')
    test  = test.merge(cc_agg,  on='SK_ID_CURR', how='left')
    del cc_agg
    gc.collect()

    pos_agg = aggregate_pos_cash(data_dir)
    train = train.merge(pos_agg, on='SK_ID_CURR', how='left')
    test  = test.merge(pos_agg,  on='SK_ID_CURR', how='left')
    del pos_agg
    gc.collect()

    print(f"\n=== FINAL FEATURE SET ===")
    print(f"  Train shape: {train.shape}")
    print(f"  Test shape:  {test.shape}")

    return train, test


if __name__ == '__main__':
    import os
    from src.data_loader import load_main_tables

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

    app_train, app_test = load_main_tables(DATA_DIR)
    train, test = build_features(app_train, app_test, DATA_DIR)

    print("\nfeature_engineering.py working correctly")