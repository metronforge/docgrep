import pytest

from docgrep.domain.errors import (
    ConfigDrift,
    DocgrepError,
    IndexLockHeld,
    ModelUnavailable,
    StoreUnavailable,
    ValidationError,
)


@pytest.mark.parametrize(
    ("error_type", "category"),
    [
        (ValidationError, "validation_error"),
        (ModelUnavailable, "model_unavailable"),
        (StoreUnavailable, "store_unavailable"),
        (ConfigDrift, "config_drift"),
        (IndexLockHeld, "index_lock_held"),
    ],
)
def test_structured_errors(error_type: type[DocgrepError], category: str) -> None:
    error = error_type("Something failed", details={"source_id": "source-1"})

    assert str(error) == "Something failed"
    assert error.category == category
    assert error.details == {"source_id": "source-1"}
