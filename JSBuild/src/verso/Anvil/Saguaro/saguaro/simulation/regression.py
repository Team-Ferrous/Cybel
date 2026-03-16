"""Utilities for regression."""

class RegressionPredictor:
    """Predicts if a change to a subsystem will cause regressions based on historical data.
    "Changes to this subsystem historically break X.".
    """

    def predict_regression(self, changed_files: list[str]) -> list[str]:
        """Returns a list of potentially broken test suites or modules."""
        risks = []
        for f in changed_files:
            if "auth" in f:
                risks.append("tests/test_login.py")
        return risks
