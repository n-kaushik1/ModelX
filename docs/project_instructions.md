# DA5401 Data Analytics Lab — Course Project

## 1. Project Objective

The objective of this project is to build a **reusable end-to-end machine learning framework** for binary classification problems.

You are given three public classification problems from different industry contexts. Each problem has different data characteristics and requires different modelling decisions.

Your task is **not** to build three unrelated one-off models. You should develop a common ML framework that can:

- inspect the supplied training data;
- identify relevant data characteristics;
- construct appropriate preprocessing;
- compare suitable modelling approaches;
- perform model selection and limited hyperparameter tuning;
- train the selected model;
- generate probability predictions;
- and, where required, determine an appropriate classification threshold.

The same framework will later be evaluated on unseen classification problems.

---

## 2. Public Evaluation Structure

For each public problem, you are given a **development dataset** containing:

- input features;
- the binary target;
- a `record_id` column used only as an identifier.

A separate **leaderboard holdout set** exists for each problem.

The holdout rows and labels are not provided to students.

When the public leaderboard is active, your submitted framework will be trained using the corresponding development dataset and evaluated automatically on the hidden holdout set.

Therefore:

> The leaderboard should be used as an external check of generalization, not as a substitute for your own validation strategy.

---

# 3. Public Problem 1 — Manufacturing Batch Quality Prediction

## Scenario

A manufacturing plant produces multiple product families using different raw materials, suppliers, production lines, machines and operating conditions.

Every batch undergoes final quality inspection. The plant would like to estimate, using information available during production, whether a batch is likely to fail the final quality inspection.

Your task is to estimate the probability of quality failure.

## Target

`quality_failure`

- `0` — batch passes final quality inspection
- `1` — batch fails final quality inspection

## Primary Evaluation Metric

**ROC-AUC**

## Dataset File

`p1_manufacturing_development.csv`

---

# 4. Public Problem 2 — SME Credit Delinquency Probability

## Scenario

A financial institution provides working-capital loans to small and medium businesses.

The institution wants to estimate the probability that a borrower will become seriously delinquent during the defined future performance period.

The resulting probability estimate may subsequently be used for credit decisioning, pricing, limits and portfolio monitoring.

The objective is therefore to produce **good probability estimates**, not simply class labels.

## Target

`serious_delinquency`

- `0` — no serious delinquency
- `1` — serious delinquency

## Primary Evaluation Metric

**Log Loss**

For binary classification:

\[
\text{Log Loss}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
\]

where \(p_i\) is the predicted probability of the positive class.


## Dataset File

`p2_credit_development.csv`

---

# 5. Public Problem 3 — Telecom Network Fault Escalation

## Scenario

A telecommunications operator continuously receives alarms and performance indicators from network nodes.

Most alarms are resolved without major customer impact. A relatively small fraction escalate into significant service disruptions requiring intervention.

Operations teams cannot investigate every alert. Missing an event that later escalates is substantially more costly than investigating an alert that ultimately proves benign.

Your framework must therefore:

1. estimate the probability that an event will escalate; and
2. determine an appropriate decision threshold for intervention.

## Target

`major_service_disruption`

- `0` — event does not escalate
- `1` — event escalates into a major service disruption

## Decision Costs

For this problem:

- False Positive cost = **1**
- False Negative cost = **10**

For a set of predictions:

\[
\text{Cost}
=
1 \times FP
+
10 \times FN
\]

Your framework should determine the decision threshold using only the available development data.

The optimal threshold is **not provided**.

## Primary Evaluation Metric

The leaderboard will use a normalized decision-cost score derived from the above cost function.

Lower raw decision cost is better.

## Dataset File

`p3_telecom_development.csv`

---

# 6. Important Dataset Rules

## `record_id`

Each dataset contains a `record_id` column.

`record_id` is an identifier and **must not be used as a predictive feature**.

Your framework should work even if identifier columns are absent in future datasets.

## Target Column

The target column is included only in the development datasets.

Do not assume that the target column will appear in test data.

## Feature Schema

Do not assume:

- fixed feature names across problems;
- a fixed number of features;
- a fixed ordering of columns;
- that all features are numerical;
- that all categorical variables contain the same categories in training and test data;
- that missing values are absent;
- that all numerical variables have similar scale.

Your framework should be robust to such differences.

---

# 7. Expected ML Workflow

A strong submission is expected to implement a workflow that can perform appropriate steps such as:

1. inspect the supplied training data;
2. separate identifiers, features and target;
3. identify numerical and categorical variables;
4. handle missing values;
5. encode categorical variables when required;
6. scale or transform features where appropriate;
7. establish one or more baseline models;
8. compare suitable candidate models;
9. use an appropriate validation strategy;
10. perform limited hyperparameter tuning;
11. select a final model based on validation evidence;
12. fit the selected model using the available training data;
13. generate positive-class probability predictions;
14. determine a decision threshold where the problem requires one.

Not every technique is required for every dataset. The framework should make decisions based on the supplied data and problem objective.

---

# 8. Reusable Framework Requirement

You should build **one reusable framework**, rather than three independently hard-coded solutions.

The final evaluation interface will follow the general form:

```python
class StudentSolution:

    def fit(self, X, y, metadata):
        ...

    def predict_proba(self, X, metadata):
        ...

    def get_threshold(self):
        ...
```

Your implementation may contain additional helper functions and classes.

The framework may adapt its behaviour using:

- the supplied training data;
- measured data characteristics;
- the evaluation objective supplied through `metadata`.

It must not identify or special-case a known public dataset using:

- specific feature names;
- row counts;
- column counts;
- dataset-specific values;
- or other dataset fingerprints.

The same framework will later be applied to previously unseen classification problems.

---

# 9. Allowed Modelling Approaches

You may use techniques covered in the course and standard tabular ML methods, including:

- Logistic Regression;
- Decision Trees;
- Random Forests;
- LightGBM;
- XGBoost;
- CatBoost;
- HistGradientBoosting;
- scikit-learn ensemble classsifiers;
- ensembles of above permitted methods;
- preprocessing pipelines;
- imputation;
- encoding;
- scaling;
- feature transformations;
- feature selection;
- class weighting;
- resampling;
- probability calibration;
- cross-validation;
- Grid Search; 
- Randomized Search;
- Optuna based Bayesian search. 


You are expected to choose techniques based on evidence rather than apply every technique indiscriminately.

---

# 10. Not Permitted in the Final Framework

The final executable framework must not depend on:

- AutoML systems that automatically construct the entire modelling solution;
- neural-network or deep-learning frameworks for this project;
- external pretrained models;
- internet access;
- external APIs;
- external LLM calls;
- downloading additional datasets or models during evaluation;
- stored predictions;
- pre-fitted models specific to the public datasets;
- dataset-specific hard-coded rules.

AI tools may be used during development for learning, coding or debugging, but you remain responsible for understanding and validating your submitted solution.

---

# 11. Validation and Leaderboard Use

You should develop and evaluate your framework locally using the released development datasets.

Suitable approaches may include:

- train/validation splits;
- stratified cross-validation;
- out-of-fold predictions;
- model comparison using the problem's primary metric.

Do not assume that a small improvement on the public leaderboard necessarily represents a genuine improvement in generalization.

Repeatedly changing a model solely in response to leaderboard scores can lead to leaderboard overfitting. Public submission limits will therefore be applied.

---

# 12. Public and Private Evaluation

The project has two evaluation stages.

## Public Evaluation

Your framework will be evaluated on hidden holdout observations corresponding to the three public problems.

Public scores will be visible during development.

## Private Evaluation

After the final deadline, your submitted framework will be frozen.

The same framework will then be trained and evaluated automatically on **three additional hidden binary classification problems**.

These problems will assess whether your methodology generalizes beyond the public datasets.

No changes to the framework will be allowed after the submission deadline.

---

# 13. Generalization Is Part of the Project

The goal is not to discover the single best algorithm for the three released datasets.

A strong framework should be able to answer questions such as:

- What type of data am I receiving?
- What preprocessing is required?
- What validation strategy is appropriate?
- Which model families are worth comparing?
- How should model complexity be controlled?
- Is class imbalance important?
- Does the evaluation objective depend on probabilities, ranking or final decisions?
- How should a threshold be chosen when decision costs matter?

Your private evaluation score will depend on how well these principles transfer to unseen problems.

---

# 14. Final Brief Report

After the leaderboard is frozen, each student must submit a **brief final report** using the template provided separately.

The report should summarize the reasoning behind the final framework and should not be a detailed project report.

It should include, for each public problem:

- key characteristics identified in the dataset;
- validation strategy used;
- preprocessing decisions;
- models/approaches compared;
- final modelling approach selected;
- important tuning decisions;
- threshold-selection approach for P3;
- one or two key observations from the experiments;

The report should also include a short overall reflection on:

- how the common framework adapts across the three problems;
- what aspects of the framework are expected to generalize to unseen private problems;
- and any significant changes made to the framework based on public leaderboard feedback.

Keep the report concise and focused on **decisions and reasoning**, rather than including extensive code, screenshots, plots, or repeated leaderboard results.

---

Additional submission templates and leaderboard instructions will be provided separately.
