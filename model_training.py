# model_training.py

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os

def hyperparameter_optimization(model, param_grid, X_train, y_train):
    """
    Perform hyperparameter optimization using GridSearchCV for a given model.
    """
    multi_model = MultiOutputClassifier(model, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=multi_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_micro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_models(X_train, y_train):
    """
    Train and optimize base models, then combine them into a VotingClassifier.
    """
    # Define Base Models
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    lgbm = LGBMClassifier(random_state=42, verbose=-1)
    svc = SVC(probability=True, random_state=42)

    # Define Parameter Grids
    rf_param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [None, 10],
        'estimator__min_samples_split': [2, 5]
    }

    gb_param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__max_depth': [3, 5]
    }

    xgb_param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__max_depth': [3, 5],
        'estimator__subsample': [0.8, 1]
    }

    lgbm_param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__num_leaves': [31, 50],
        'estimator__max_depth': [-1, 10],
        'estimator__min_child_samples': [10, 20],
        'estimator__min_split_gain': [0.0],
        'estimator__max_bin': [255]
    }

    svc_param_grid = {
        'estimator__C': [1, 10],
        'estimator__kernel': ['linear', 'rbf']
    }

    # Optimize Each Base Model
    print("Optimizing RandomForestClassifier...")
    best_rf = hyperparameter_optimization(rf, rf_param_grid, X_train, y_train)

    print("Optimizing GradientBoostingClassifier...")
    best_gb = hyperparameter_optimization(gb, gb_param_grid, X_train, y_train)

    print("Optimizing XGBClassifier...")
    best_xgb = hyperparameter_optimization(xgb, xgb_param_grid, X_train, y_train)

    print("Optimizing LGBMClassifier...")
    best_lgbm = hyperparameter_optimization(lgbm, lgbm_param_grid, X_train, y_train)

    print("Optimizing SVC...")
    best_svc = hyperparameter_optimization(svc, svc_param_grid, X_train, y_train)

    # Initialize Voting Classifier with Optimized Estimators
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf.estimators_[0]),
            ('gb', best_gb.estimators_[0]),
            ('xgb', best_xgb.estimators_[0]),
            ('lgbm', best_lgbm.estimators_[0]),
            ('svc', best_svc.estimators_[0])
        ],
        voting='soft',
        n_jobs=-1
    )

    # Wrap Voting Classifier in MultiOutputClassifier
    multi_voting_clf = MultiOutputClassifier(voting_clf, n_jobs=-1)

    # Fit the Model
    print("\nTraining the Voting Classifier...")
    multi_voting_clf.fit(X_train, y_train)

    return multi_voting_clf

def evaluate_model(model, X_test, y_test, y_columns):
    """
    Evaluate the trained model on the test set.
    """
    print("\nEvaluating the Voting Classifier...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=y_columns))


def save_model(model, tfidf_vectorizer, scaler):
    """
    Save the trained model and preprocessing objects using joblib.
    Saves the files in the 'models' directory.
    """
    # Create 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model and preprocessing objects in the 'models' directory
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')