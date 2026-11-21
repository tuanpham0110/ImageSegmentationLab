from pathlib import Path, PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

from scripts.package_submission import ARCHIVE_ROOT, REQUIRED_FILES, ROOT, package_submission


def test_submission_contains_demo_and_acceptance_without_local_artifacts(tmp_path):
    output = tmp_path / "submission.zip"
    count = package_submission(ROOT, output)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        paths = [PurePosixPath(name).relative_to(ARCHIVE_ROOT) for name in archive.namelist()]
        files = {path.as_posix() for path in paths}
        assert count == len(paths) == len(files)
        assert {"app.py", "README.md", "requirements.txt", "requirements-dev.txt",
                "data/samples.json", "docs/REPORT.md", "docs/SLIDES.pptx", "docs/RELEASE.md",
                "tests/browser_acceptance.cjs", "tests/verify_acceptance_exports.py"} <= files
        assert not any(part in {".git", "outputs", "external", "__pycache__", ".pytest_cache", "secrets.toml"}
                       or part.startswith(".venv") for path in paths for part in path.parts)
        for name in sorted(files):
            if name.startswith("docs/results/") and name.endswith(".xml"):
                report = ElementTree.fromstring(archive.read(f"{ARCHIVE_ROOT}/{name}"))
                assert all("hostname" not in suite.attrib for suite in report.iter("testsuite"))

    with pytest.raises(FileExistsError):
        package_submission(ROOT, output)


def test_missing_slide_stops_build_before_creating_zip(tmp_path):
    missing = "docs/SLIDES.pptx"
    root = tmp_path / "source"
    for name in REQUIRED_FILES:
        if name == missing:
            continue
        path = root / Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    output = tmp_path / "submission.zip"
    with pytest.raises(FileNotFoundError, match=r"docs/SLIDES\.pptx"):
        package_submission(root, output)
    assert not output.exists()
