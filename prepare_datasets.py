import numpy as np
import pandas as pd


def get_and_prepare_german_dataset():

    X = pd.read_csv("data/german_processed.csv")
    y = X["GoodCustomer"]
    # Store feature names before dropping columns
    feature_names = list(X.columns)
    feature_names.remove("GoodCustomer")
    feature_names.remove("PurposeOfLoan")
    feature_names.remove("OtherLoansAtStore")

    # we remove the features that are not useful for our analysis
    # we also remove 'OtherLoansAtStore' because it has only has one value 0
    X = X.drop(["GoodCustomer", "PurposeOfLoan","OtherLoansAtStore"], axis=1)

    # we want our data to be numerical so Male -> 0, Female -> 1
    X['Gender'] = [0 if v == "Male" else 1 for v in X['Gender'].values]

    # y.values is either 1 or -1, so 1 stays 1 and -1 becomes 0
    y = np.array([1 if p == 1 else 0 for p in y.values])

    categorical_features = [
        'Gender', 'ForeignWorker', 'Single', 'HasTelephone',
        'CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200',
        'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500',
        'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere',
        'OtherLoansAtBank', 'HasCoapplicant',
        'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed',
        'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled'
    ]

    continuous_features = [
        'Age', 'LoanDuration', 'LoanAmount', 'LoanRateAsPercentOfIncome',
        'YearsAtCurrentHome', 'NumberOfOtherLoansAtBank', 'NumberOfLiableIndividuals']

    actionable_features = ['LoanDuration', 'LoanAmount', 'LoanRateAsPercentOfIncome', 'CheckingAccountBalance_geq_0',
                           'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100',
                           'SavingsAccountBalance_geq_500', 'MissedPayments', 'NumberOfOtherLoansAtBank',
                           'OtherLoansAtBank', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse',
                           'RentsHouse', 'Unemployed', 'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4']


    
    return X.values, y , feature_names , categorical_features , continuous_features, actionable_features
