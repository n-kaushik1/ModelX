"""The three public problems as the leaderboard defines them (used by make_submission.py and local_test.py).
Your framework must not use this file: it should work from metadata alone."""

PROBLEMS = [
    {"key": "p1", "file": "p1_manufacturing_development.csv", "target": "quality_failure", "objective": "roc_auc"},
    {"key": "p2", "file": "p2_credit_development.csv", "target": "serious_delinquency", "objective": "log_loss"},
    {"key": "p3", "file": "p3_telecom_development.csv", "target": "major_service_disruption",
     "objective": "decision_cost"},
]


def metadata_for(objective):
    cost = objective == "decision_cost"
    return {
        "objective": objective,
        "greater_is_better": objective == "roc_auc",
        "requires_threshold": cost,
        "cost_false_positive": 1 if cost else None,
        "cost_false_negative": 10 if cost else None,
        "positive_label": 1,
    }
