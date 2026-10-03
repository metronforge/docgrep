"""The embedding dimension is read with the accessor the model provides."""

import pytest

from docgrep.adapters.models.real import BgeEmbedder
from docgrep.domain.errors import ModelUnavailable


class _Tokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": []}


class OldNameOnly:
    tokenizer = _Tokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return 384


class NewNameOnly:
    tokenizer = _Tokenizer()

    def get_embedding_dimension(self) -> int:
        return 768


class BothNames:
    """Mimics a release where the old name still exists but is deprecated."""

    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.old_called = False

    def get_embedding_dimension(self) -> int:
        return 1024

    def get_sentence_embedding_dimension(self) -> int:
        self.old_called = True
        return -1


class NoAccessor:
    tokenizer = _Tokenizer()


class NoneDimension:
    tokenizer = _Tokenizer()

    def get_embedding_dimension(self) -> int | None:
        return None


def _dimension(model: object) -> int:
    return BgeEmbedder("m", "cpu", lambda *_: model).dimension


@pytest.mark.parametrize(
    ("model", "expected"),
    [(OldNameOnly(), 384), (NewNameOnly(), 768), (BothNames(), 1024)],
)
def test_dimension_uses_whichever_accessor_exists(model: object, expected: int) -> None:
    assert _dimension(model) == expected


def test_new_name_is_preferred_so_the_deprecated_one_is_never_called() -> None:
    model = BothNames()

    _dimension(model)

    assert model.old_called is False


@pytest.mark.parametrize("model", [NoAccessor(), NoneDimension()])
def test_missing_accessor_or_dimension_is_a_model_error(model: object) -> None:
    with pytest.raises(ModelUnavailable) as raised:
        _dimension(model)

    assert raised.value.details["model"] == "m"
