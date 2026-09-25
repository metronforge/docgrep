import re
from importlib.metadata import version

import docgrep


def test_package_exposes_semantic_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", docgrep.__version__)
    assert docgrep.__version__ == version("docgrep")
