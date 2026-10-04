"""Soft hyphens and the garbled text layer rules, on plain strings (UC-3)."""

import pytest

from docgrep.adapters.parsers.text_quality import (
    LONG_TOKEN_LENGTH,
    MIN_NON_WHITESPACE,
    is_garbled,
    remove_soft_hyphens,
)

SHY = "\u00ad"


def test_a_soft_hyphen_inside_a_line_is_removed() -> None:
    assert remove_soft_hyphens(f"hyphen{SHY}ation works") == ("hyphenation works", 1)


def test_a_soft_hyphen_ending_a_line_joins_the_word() -> None:
    text = f"the hyphen{SHY}\nation of words\n"

    assert remove_soft_hyphens(text) == ("the hyphenation of words\n", 1)


def test_spaces_after_a_line_end_soft_hyphen_go_with_the_break() -> None:
    assert remove_soft_hyphens(f"hyphen{SHY}   \r\nation") == ("hyphenation", 1)


def test_the_russian_word_is_joined() -> None:
    text = f"Это пример пере{SHY}\nноса слов{SHY}а в тексте.\n"

    assert remove_soft_hyphens(text) == ("Это пример переноса слова в тексте.\n", 2)


def test_a_visible_hyphen_and_everything_else_stay_verbatim() -> None:
    text = "well-known co-\noperation  with  spaces\n\n\tand tabs \u00a0end\n"

    assert remove_soft_hyphens(text) == (text, 0)


def _page(lines: list[str]) -> str:
    text = "\n".join(lines) + "\n"
    assert sum(not c.isspace() for c in text) >= MIN_NON_WHITESPACE
    return text


NOT_GARBLED = {
    "english prose": _page(
        [
            "The survey of the northern valley began in the spring, when the river",
            "was still high and the roads were difficult. The team recorded water",
            "levels twice a day and collected stones from every bend of the stream.",
            "Their notes, letters and maps were kept in the archive for decades.",
        ]
    ),
    "russian prose": _page(
        [
            "Экспедиция в северную долину началась весной, когда река ещё была",
            "высокой, а дороги трудными. Участники дважды в день записывали уровень",
            "воды и собирали камни на каждом изгибе реки. Их заметки, письма и",
            "карты десятилетиями хранились в архиве краеведческого музея.",
        ]
    ),
    "source code": _page(
        [
            "def build_reverse_dependency_index_for_all_registered_packages(self):",
            "    result: dict[str, list[str]] = {}",
            "    for name, spec in self._registry.items():  # walk every package",
            "        for dep in spec.requires_dist_including_optional_extras():",
            "            result.setdefault(dep.normalized_name, []).append(name)",
            "    if not result or len(result) > MAX_INDEX_SIZE_BEFORE_COMPACTION:",
            "        raise IndexOverflowError(f'{len(result)=} > {MAX=}')",
            "    return {k: sorted(v) for k, v in result.items() if v != []}",
        ]
    ),
    "table of numbers": _page(
        [
            "Year   Q1        Q2        Q3        Q4        Total     Change",
            "2019   1,204.50  1,310.25  1,287.00  1,402.75  5,204.50  +3.2%",
            "2020   1,150.00  980.40    1,045.90  1,230.10  4,406.40  -15.3%",
            "2021   1,301.20  1,355.80  1,420.00  1,512.35  5,589.35  +26.8%",
            "2022   1,498.00  1,520.10  1,610.45  1,702.90  6,331.45  +13.3%",
        ]
    ),
    "page of urls": _page(
        [
            "https://www.example.org/archive/collections/northern-valley-survey-1923/maps",
            "https://docs.example.com/reference/api/v2/endpoints/search?query=river&page=3",
            "www.example.net/letters/expedition/1924/correspondence/volume-two/index.html",
            "http://library.example.edu/catalogue/records/item/000123456789/full-record",
        ]
    ),
    "bibliography": _page(
        [
            "[1] Smith, J., & Ivanova, M. (2019). River sediment transport in",
            "    northern valleys. Journal of Hydrology, 571, 123-145.",
            "    doi:10.1016/j.jhydrol.2019.01.042",
            "[2] Петров, А. В. (2015). Гидрология малых рек. Москва: Наука. 312 с.",
            "[3] Brown, K. (2021). Field Methods. 2nd ed. Oxford UP, pp. 45-67.",
        ]
    ),
}


@pytest.mark.parametrize("name", sorted(NOT_GARBLED))
def test_readable_pages_are_never_garbled(name: str) -> None:
    assert not is_garbled(NOT_GARBLED[name])


def test_a_short_page_is_never_judged() -> None:
    assert not is_garbled("CHAPTER ONE\n" + "\ufffd" * 50 + "\n")
    assert not is_garbled("#" * (MIN_NON_WHITESPACE - 1))


GARBLED = {
    "upper-case run-on": "\n".join(
        "QWERTYUIOPASDFGHJKLZXCVBNMQWERTYUIOPASDFGHJKLZXCVBNMQWERTYUIOPASDFGH"
        for _ in range(5)
    ),
    "symbol soup": "\n".join(
        "#$% &*( )_+ {}| :<> ?~` !@ ^-= [];' ,./" for _ in range(8)
    ),
    "replacement characters": "\n".join(
        "the river \ufffd\ufffd\ufffd survey \ufffd\ufffd "
        "notes \ufffd\ufffd\ufffd\ufffd maps"
        for _ in range(8)
    ),
    "cid patterns": " ".join("(cid:12)(cid:345) word" for _ in range(20)),
    "private use": "\n".join(
        "valley \ue001\ue002\ue003 stones \uf8ff\ue010 water \ue020\ue021"
        for _ in range(9)
    ),
}


@pytest.mark.parametrize("name", sorted(GARBLED))
def test_garbled_pages_are_flagged(name: str) -> None:
    assert sum(not c.isspace() for c in GARBLED[name]) >= MIN_NON_WHITESPACE
    assert is_garbled(GARBLED[name])


def test_long_urls_are_not_long_tokens() -> None:
    url = "https://example.org/" + "a" * (LONG_TOKEN_LENGTH * 2)
    page = "\n".join([url] * 3 + ["www." + "b" * 100] * 2)

    assert not is_garbled(page)
