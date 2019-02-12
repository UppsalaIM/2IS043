# -*- coding: utf-8 -*-
import pandas as pd
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    widgets
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn import tree

body_stats = pd.read_csv("weight-height.csv")
titanic_passengers = pd.read_csv("titanic.csv")

def view_body_stats_table():
    global body_stats
    return body_stats

def fix_body_stats_units():
    global body_stats
    body_stats.Height = body_stats.Height.apply(lambda x: x * 2.54)
    body_stats.Weight = body_stats.Weight.apply(lambda x: x / 2.2046)
    return body_stats

def plot_body_stats_scatter():
    _ = body_stats.plot.scatter(x='Weight', y='Height')

def body_stats_scatter_by_colour(m_colour, f_colour):
    cmap = {'Male': m_colour, 'Female': f_colour}
    _ = body_stats.plot.scatter(x='Weight', y='Height', c=[cmap.get(c) for c in body_stats.Gender])

def plot_body_stats_scatter_by_colour():
    global body_stats
    colours_list = [
        'blue',
        'green',
        'red',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'white'
    ]
    interact(body_stats_scatter_by_colour, m_colour=widgets.Dropdown(
        options=colours_list,
        value='blue',
        description='Male:',
        disabled=False,
    ), f_colour=widgets.Dropdown(
        options=colours_list,
        value='blue',
        description='Female:',
        disabled=False,
    ))

def plot_interact_scatter(intercept, slope, category, colour):
    global body_stats
    fig, ax = plt.subplots()
    stats = body_stats[body_stats.Gender==category]
    plt.scatter(x=stats.Weight, y=stats.Height, c=colour)
    predicted_values_m = [slope * i + intercept for i in stats.Weight]
    plt.plot(stats.Weight, predicted_values_m, '--')
    plt.xlabel("Weight")
    plt.ylabel("Height")

def plot_interactive_body_stats_scatter(category, colour):
    interact(plot_interact_scatter,
         intercept=widgets.IntSlider(value=1, min=-300, max=300, continuous_update=False),
         slope=widgets.FloatSlider(value=3, min=0, max=10.0, continuous_update=False),
         category=fixed(category), colour=fixed(colour))

def linear_regression(category):
    global body_stats
    stats = body_stats[body_stats.Gender==category]
    lm = linear_model.LinearRegression()
    lm.fit([[x] for x in stats.Weight], stats.Height)
    m = lm.coef_[0]
    b = lm.intercept_
    print("slope=", m, "intercept=", b)
    return lm

def plot_linear_regression_on_scatter(category, colour):
    global body_stats
    stats = body_stats[body_stats.Gender==category]
    lm = linear_regression(category)
    m = lm.coef_[0]
    b = lm.intercept_
    plt.scatter(stats.Weight, stats.Height, c=colour)
    predicted_values_f = [lm.coef_ * i + lm.intercept_ for i in stats.Weight]
    plt.plot(stats.Weight, predicted_values_f, '--')
    plt.ylabel("Height")
    plt.xlabel("Weight")

def view_titanic_table():
    global titanic_passengers
    return titanic_passengers

def titanic_value_counts(column):
    return view_titanic_table()[column].value_counts()

def plot_titanic_value_counts(column):
    _ = titanic_value_counts(column).plot.bar()

def plot_titanic_survived_by_gender():
    survived_by_gender = view_titanic_table().groupby('sex').survived.mean()
    _ = survived_by_gender.plot.bar()
    return survived_by_gender

def plot_titanic_survived_by_age():
    survived_by_age = view_titanic_table().groupby('age').survived.sum()
    _ = survived_by_age.plot.bar()
    return survived_by_age

def view_interactive_titanic_table():
    def view_table_with_ages(child_age):
        titanic_passengers = view_titanic_table()
        titanic_passengers['age_range'] = pd.cut(titanic_passengers.age, [0, child_age, 80], labels=['child', 'adult'])
        return titanic_passengers
    interact(view_table_with_ages, child_age=widgets.IntSlider(value=18, min=min(titanic_passengers.age) + 1, max=max(titanic_passengers.age) - 1, continuous_update=False))

def view_interactive_titanic_bar():
    def view_table_with_ages(child_age):
        titanic_passengers = view_titanic_table().copy()
        titanic_passengers['age_range'] = pd.cut(titanic_passengers.age, [0, child_age, 80], labels=['child', 'adult'])
        survived_by_age = titanic_passengers.groupby('age_range').survived.mean()
        _ = survived_by_age.plot.bar(ylim=(0,1))
    interact(view_table_with_ages, child_age=widgets.IntSlider(value=18, min=min(titanic_passengers.age) + 1, max=max(titanic_passengers.age) - 1, continuous_update=False))

def impute_numeric(method, column):
    if method == 'median':
        return pd.DataFrame({
            "original": view_titanic_table()[column].describe(),
            "imputed": view_titanic_table()[column].fillna(view_titanic_table()[column].median()).describe()
        })
    elif method == 'mean':
        return pd.DataFrame({
            "original": view_titanic_table()[column].describe(),
            "imputed": view_titanic_table()[column].fillna(view_titanic_table()[column].mean()).describe()
        })
    else:
        return pd.DataFrame({
            "original": view_titanic_table()[column].describe(),
            "imputed": view_titanic_table()[column].fillna(np.random.uniform(view_titanic_table()[column].max())).describe()
        })

def titanic_impute_numeric(column):
    interact(impute_numeric, method=widgets.RadioButtons(
        options=['mean', 'median', 'random'],
        description='Method:',
        disabled=False),
        column=fixed(column)
    )

def view_titanic_sex_male():
    titanic_passengers = view_titanic_table()
    titanic_passengers.age = titanic_passengers.age.fillna(titanic_passengers.age.mean())
    titanic_passengers.fare = titanic_passengers.fare.fillna(titanic_passengers.fare.mean())
    titanic_passengers = pd.get_dummies(titanic_passengers, columns=['sex'], drop_first=True)
    return titanic_passengers

def view_titanic_final_table():
    return view_titanic_sex_male()[['sex_male', 'fare', 'age', 'sibsp', 'survived']]

def split_datasets(test_size_percent):
    test_size_fraction = test_size_percent / 100
    titanic_passengers = view_titanic_final_table()[['fare', 'age', 'sibsp', 'sex_male']]
    survived_data = view_titanic_final_table().survived
    X_train, X_test, y_train, y_test = train_test_split(titanic_passengers, survived_data, test_size=test_size_fraction)
    print("Our training data has {} rows".format(len(X_train)))
    print("Our test data has {} rows".format(len(X_test)))

def interact_split_datasets():
    interact(split_datasets, test_size_percent=widgets.IntSlider(value=25, min=1, max=99, continuous_update=False))

def titantic_decision_tree():
    titanic_passengers = view_titanic_final_table()[['fare', 'age', 'sibsp', 'sex_male']]
    survived_data = view_titanic_final_table().survived
    X_train, X_test, y_train, y_test = train_test_split(titanic_passengers, survived_data, test_size=0.25)
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train.values, y_train.values)
    sample = X_test
    sample['predicted_survived'] = classifier.predict(sample)
    sample['true survived'] = y_test.values
    return sample

def titantic_decision_tree_accuracy():
    titanic_passengers = view_titanic_final_table()[['fare', 'age', 'sibsp', 'sex_male']]
    survived_data = view_titanic_final_table().survived
    X_train, X_test, y_train, y_test = train_test_split(titanic_passengers, survived_data, test_size=0.25)
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train.values, y_train.values)
    sample = X_test
    sample['predicted_survived'] = classifier.predict(sample)
    sample['true survived'] = y_test.values
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test.values, classifier.predict(sample))

def split_datasets_for_dtviz(test_size_percent):
    test_size_fraction = test_size_percent / 100
    titanic_passengers = view_titanic_final_table()[['fare', 'age', 'sibsp', 'sex_male']]
    survived_data = view_titanic_final_table().survived
    X_train, X_test, y_train, y_test = train_test_split(titanic_passengers, survived_data, test_size=test_size_fraction)
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train.values, y_train.values)
    tree_plot = Source(tree.export_graphviz(classifier, out_file=None,
                            feature_names=X_train.columns, class_names=['Dead', 'Alive'],
                            filled=True, rounded=True, special_characters=True))
    return tree_plot

def plot_decision_tree():
    interact(split_datasets_for_dtviz, test_size_percent=widgets.IntSlider(value=25, min=1, max=99, continuous_update=False))
