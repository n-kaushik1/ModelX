DA5401 Course Project: Public Leaderboard
Only accounts registered with smail would be allowed

Build one reusable machine-learning framework for binary classification. The full brief is in the course project instructions; this site runs the public leaderboard.

You train your framework on each problem's development dataset and upload the trained models together with your code. The leaderboard applies your saved models to each problem's hidden holdout set and scores the predictions:

Problem	Target	Leaderboard metric
P1 Manufacturing batch quality	quality_failure	ROC-AUC (higher is better)
P2 SME credit delinquency	serious_delinquency	Log loss (lower is better)
P3 Telecom fault escalation	major_service_disruption	Normalized decision cost (lower is better)
Key dates
Leaderboard submissions close 28 November, 23:59 IST.
Final report due 29 November.
No extensions.
Results can take longer near the deadline, when many students submit at once. Submit early.
Getting started
Download the starting kit (Files / Get Started tab).
Read the Submission page.
Build your zip with make_submission.py.
Use the leaderboard to check whether your framework generalizes, not as a substitute for your own validation.
Evaluation
Your uploaded models are applied to the hidden holdout set of each problem. The holdout features and labels are never shown to you.

P1: ROC-AUC (higher is better)
Computed from the probabilities your P1 model returns.

P2: Log loss (lower is better)
Log Loss
=
−
1
N
∑
i
=
1
N
[
y
i
log
⁡
(
p
i
)
+
(
1
−
y
i
)
log
⁡
(
1
−
p
i
)
]
Log Loss=− 
N
1
​
  
i=1
∑
N
​
 [y 
i
​
 log(p 
i
​
 )+(1−y 
i
​
 )log(1−p 
i
​
 )]

Probabilities are clipped to [10-15, 1 - 10-15] before scoring. Confident wrong predictions are heavily penalized.

P3: Normalized decision cost (lower is better)
Your P3 model returns probabilities, and your p3_threshold.json gives the threshold. A case counts as an alert when probability >= threshold.

Raw decision cost on the holdout set:

Cmodel = FP + 10 × FN

FP = number of false positives
FN = number of false negatives
The holdout set must play no role in choosing the threshold.

Baseline cost. The model is compared against the better of two trivial strategies:

Predict every case negative: Call-0 = 10 N+, where N+ is the number of actual positives in the holdout.
Predict every case positive: Call-1 = N-, where N- is the number of actual negatives in the holdout.
Cbaseline = min(10 N+, N-)

Normalized decision cost = Cmodel / Cbaseline

0 = perfect decisions
1 = as good as the best trivial strategy
below 1 = better than the best trivial strategy
above 1 = worse than the best trivial strategy
Leaderboard
You choose which of your successful submissions is shown, on the My Submissions tab.
The submission shown on the leaderboard at the deadline is treated as your final submission.
Average rank is the mean of your ranks on P1, P2 and P3, shown for orientation.


Submission
You train your framework on your own computer and upload your code together with the trained models. The leaderboard does not train anything: it loads your saved models and predicts the hidden holdout sets.

⚠️ Put your submission on the leaderboard (required)
Finished submissions are not added to the leaderboard automatically.

Go to the My Submissions tab.
Wait until your submission shows Finished.
In the Actions column of the submission you want to show, click the green leaderboard icon.
Open the Results tab and check that your entry is there.
You can switch to a different submission at any time before the deadline.

The submission shown on the leaderboard when it closes (28 Nov, 11:59 PM IST) is your final submission. If no submission is shown, you have no leaderboard entry.

After every submission, open Detailed Results (eye icon) to see your scores and any warnings. If a submission Failed, click on it and read the ingestion log to see what went wrong.

Wait for each upload to finish before leaving the page or uploading again. Uploading the same zip again does not make it run faster, and every upload uses one of your daily attempts.

What to upload
A .zip file with these files at the top level:

submission.zip
├── solution.py           your complete framework code (other .py helper files allowed)
├── p1_model.joblib       fitted model for P1
├── p2_model.joblib       fitted model for P2
├── p3_model.joblib       fitted model for P3
├── p3_threshold.json     {"threshold": 0.123}   (alert when probability >= threshold)
└── manifest.json         Python and library versions used for training
The starting kit's make_submission.py builds and checks this zip for you:

python make_submission.py path/to/folder_with_the_development_csvs
It trains the StudentSolution class from solution.py on each full development dataset, saves the fitted objects with joblib, writes the P3 threshold from get_threshold(), and checks that the zip loads and predicts correctly.

Rules for the saved models
Each model must include all preprocessing. The leaderboard passes the raw holdout features to your model: the same columns as the development file, without the target and record_id.
Everything from raw features to probabilities must be inside the saved object: imputation, encoding, scaling, feature engineering and the model.
Nothing is fitted on the holdout data.
Holdout data can contain category values or missing values that do not appear in the development data. Your saved preprocessing must handle them.
The leaderboard calls predict_proba(X, metadata) on your fitted StudentSolution. Plain models with predict_proba(X), such as a fitted scikit-learn pipeline, are also accepted.
predict_proba must return one positive-class probability per row, in [0, 1], with no NaN. An (n, 2) array is also accepted; column 1 is used.
Train only on the development data, with the code in solution.py. In the private evaluation, the same solution.py is retrained on unseen problems.
Environment
Models load correctly only with the same library versions they were saved with. Train in Python 3.11 with exactly these versions (requirements.txt in the starting kit):

numpy 1.26.4, pandas 2.3.1, scipy 1.17.1, scikit-learn 1.7.1, joblib 1.5.1, lightgbm 4.6.0, xgboost 3.1.1, catboost 1.2.10, optuna 4.7.0, optuna-integration 4.7.0, imbalanced-learn 0.14.0.

If your manifest.json (or scikit-learn itself) shows different versions, your submission is still scored, with a warning in the log and results. Its predictions may differ from those on your computer. If a model cannot be loaded at all, the submission fails and does not count.
No other libraries are available on the leaderboard. There is no internet access.
Limits
3 submissions per day, 30 in total.
Submissions that fail a basic check do not count. Basic checks include a missing file, a model that cannot be loaded, an error in your code, or predictions or a threshold in the wrong format.
Cancelled or deleted submissions still count.
Logs
The output your own code prints is not shown.
If a basic check fails, the log shows which problem failed, the error type, a short message and the line in your code.
make_submission.py runs the same checks locally, so most problems are caught before you upload.

Data
The development datasets are the files released with the project instructions (also in this site's public data):

p1_manufacturing_development.csv
p2_credit_development.csv
p3_telecom_development.csv
Each file contains the input features, the binary target and a record_id identifier. record_id must not be used as a feature.

The holdout sets are hidden. They come from the same setting as the development data. Your saved models should not assume that every category seen in testing also appears in training, or that missing values are absent (see Section 6 of the project instructions).

Terms
This leaderboard is for students registered in DA5401.
Your Codabench username/display name must contain your roll number (for example ee22b001 or ee22b001_name). Registrations that do not follow this would get removed
Use your own account only, and submit only your own work.
Follow the course project instructions, including Sections 8 (reusable framework, no dataset-specific rules) and 10 (not permitted in the final framework).
Your models must be produced by the submitted solution.py, trained only on the development data.
Submissions may be inspected and re-run. Breaking these rules can lead to disqualification.
