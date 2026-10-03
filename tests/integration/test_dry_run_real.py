"""Dry-run sampling with the real backend: tokenizer files only, same chunking."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from docgrep.adapters.manifest import SQLiteManifest
from docgrep.api.container import create_container, create_dry_run
from docgrep.settings import Settings

pytestmark = pytest.mark.gpu


def _corpus(root: Path) -> None:
    root.mkdir()
    words = " ".join(f"word{i} Привет мир" for i in range(700))
    (root / "a.txt").write_text(words, encoding="utf-8")


def test_dry_run_sampling_chunks_exactly_like_a_real_build(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _corpus(corpus)
    settings = Settings(
        corpus_root=corpus,
        data_dir=tmp_path / "data",
        backend="real",
        chunk_size=64,
        chunk_overlap=8,
    )

    report = create_dry_run(settings).run(sample=1).as_dict()
    estimated = report["sample"]["types"][0]["passages_per_file"]
    assert not (tmp_path / "data").exists()

    services = create_container(settings)
    summary = services.build.run()
    journal = SQLiteManifest(tmp_path / "data").run_sources(summary.run_id)

    assert [r.passages for r in journal] == [estimated]
    assert estimated > 5


def test_dry_run_sampling_loads_no_model_and_no_device(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _corpus(corpus)
    script = textwrap.dedent(
        """
        import json, sys
        from docgrep.api.cli import main
        code = main(["dry-run", "--sample", "1"])
        import torch
        print(json.dumps({
            "code": code,
            "sentence_transformers": "sentence_transformers" in sys.modules,
            "cuda_initialized": torch.cuda.is_initialized(),
        }))
        """
    )
    env = {
        "DOCGREP_CORPUS_ROOT": str(corpus),
        "DOCGREP_DATA_DIR": str(tmp_path / "data"),
        "DOCGREP_BACKEND": "real",
        "HF_HUB_OFFLINE": "1",
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    last = json.loads(result.stdout.strip().splitlines()[-1])
    assert last == {
        "code": 0,
        "sentence_transformers": False,
        "cuda_initialized": False,
    }
    assert not (tmp_path / "data").exists()
