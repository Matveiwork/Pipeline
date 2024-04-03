import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



def filter_data(df):
    df = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def remove_outliers(df):
    dfcl = df.copy()
    q25 = dfcl['year'].quantile(0.25)
    q75 = dfcl['year'].quantile(0.75)
    iqr = q75-q25
    boundaries = (q25-1.5*iqr,q75+1.5*iqr)
    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return dfcl



def create_new_features(df):
    dfcl = df.copy()
    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x
    dfcl.loc[:, 'short_model'] = dfcl['model'].apply(short_model)
    dfcl.loc[:,'age_category'] = dfcl['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return dfcl



def main():
    df = pd.read_csv('data1/homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_transformer, numerical_features),
    ('categorical', categorical_transformer, categorical_features)
    ])
    models = [
    LogisticRegression(solver='liblinear'),
    RandomForestClassifier(),
    SVC()
    ]
    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
        ('filter_columns', FunctionTransformer(filter_data)),
        ('create_new_features', FunctionTransformer(create_new_features)),
        ('remove_outliers', FunctionTransformer(remove_outliers)),
        (' preprocessor', preprocessor),
        ('classifier' , model)
        ])
        score = cross_val_score(pipe,X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean(): 4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score: .4f}')
    joblib.dump (best_pipe, 'homework30_pipe.pkl')

if __name__ == '__main__':
    main()