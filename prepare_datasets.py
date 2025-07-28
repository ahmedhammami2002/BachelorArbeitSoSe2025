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


def get_and_preprocess_cc():

    X = pd.read_csv("data/communities_and_crime.csv", index_col=0)


    y_col = 'ViolentCrimesPerPop numeric'

    # some have a missing ViolentCrimesPerPop numeric which is the crime rate these instances will be dropped
    X = X[X[y_col] != "?"]

    # other values will be float32
    X[y_col] = X[y_col].values.astype('float32')

    # get all the colums that have a missing alue
    cols_with_missing_values = []
    for col in X:
        if '?' in X[col].values.tolist():
            cols_with_missing_values.append(col)



    y = X[y_col]

    # everything over 50th percentil gets negative outcome (lots of crime is bad)
    high_violent_crimes_threshold = 50
    y_cutoff = np.percentile(y, high_violent_crimes_threshold)

    X = X.drop(cols_with_missing_values + ['communityname string', 'fold numeric', 'county numeric',
                                    'community numeric', 'state numeric'] + [y_col], axis=1)

    # setup ys
    y = np.array([0 if val > y_cutoff else 1 for val in y])

    feature_names = list(X.columns)

    actionable_features = list(X.columns)

    continuous_features = list(X.columns)

    categorical_features = []



    return X.values, y , feature_names , categorical_features , continuous_features, actionable_features




