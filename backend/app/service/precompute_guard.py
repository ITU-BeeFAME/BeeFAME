# app/service/precompute_guard.py
"""On the public demo server, slow configurations are served from the precomputed cache only."""
import os

SLOW_CLASSIFIERS = {"Support Vector Classification (SVC)"}
SLOW_METHODS = {"Exponentiated Gradient", "Grid Search"}


class NotPrecomputedError(Exception):
    pass


def check_on_demand(dataset: str, classifier: str, method: str = None) -> None:
    """Refuse to train a slow configuration on a server that sets SERVE_SLOW_FROM_CACHE_ONLY=1.

    Census Income with SVC or an in-processing method takes from minutes to hours and can exhaust
    the memory of a small server. Everywhere else (the variable unset), everything is computed.
    """
    if os.getenv("SERVE_SLOW_FROM_CACHE_ONLY") != "1":
        return
    if dataset == "adult" and (classifier in SLOW_CLASSIFIERS or method in SLOW_METHODS):
        raise NotPrecomputedError(
            "This configuration is too slow to compute on the demo server and has not been "
            "precomputed yet. Try another classifier or mitigation method, or run BeeFAME "
            "locally (see the README)."
        )
