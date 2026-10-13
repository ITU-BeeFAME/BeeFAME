# app/service/precompute_guard.py
"""The public demo server answers from the precomputed cache only and never trains a model."""
import os


class NotPrecomputedError(Exception):
    pass


def check_on_demand(dataset: str, classifier: str, method: str = None) -> None:
    """Refuse to compute a result on a server that sets SERVE_FROM_CACHE_ONLY=1.

    The demo server precomputes every configuration the wizard offers, so a miss there is either a
    configuration still being precomputed or a request the wizard never sends (other test sizes or
    hyperparameters through the API). Training it could take hours and exhaust the server's memory.
    Locally (the variable unset) every configuration is computed on demand and then cached.
    """
    if os.getenv("SERVE_FROM_CACHE_ONLY") == "1":
        raise NotPrecomputedError(
            "This configuration has not been precomputed for the demo server. Try another one, or run "
            "BeeFAME locally, where any configuration is computed on demand (see the README)."
        )
