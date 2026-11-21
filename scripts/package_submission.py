"""Create a portable ZIP of the PTIT image segmentation submission."""

from __future__ import annotations

import argparse
import json
import shutil
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "PTIT-DeTai3-ImageSegmentation"
REQUIRED_FILES = (
    ".github/workflows/ci.yml",
    ".gitignore",
    ".python-version",
    ".streamlit/config.toml",
    "README.md",
    "app.py",
    "pytest.ini",
    "requirements.in",
    "requirements.txt",
    "requirements-dev.in",
    "requirements-dev.txt",
    "ruff.toml",
    "data/samples.json",
    "docs/ACCEPTANCE.md",
    "docs/DEMO_SCRIPT.md",
    "docs/RELEASE.md",
    "docs/REPORT.md",
    "docs/SLIDES.pptx",
    "docs/TEST_CASES.md",
    "src/ui/styles.css",
    "src/ui/assets/lab-icon.svg",
    "src/ui/assets/lab-logo.svg",
    "tests/browser_acceptance.cjs",
    "tests/verify_acceptance_exports.py",
)
INCLUDED_PATTERNS = (
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/*.py",
    "data/sample_images/*.png",
    "data/sample_csv/*.csv",
    "data/ground_truth/*.png",
    "docs/*.md",
    "docs/figures/*.png",
    "docs/screenshots/*.png",
    "docs/results/**/*.json",
    "docs/results/**/*.md",
    "docs/results/**/*.xml",
)
# Keep the evidence linked from published project documents. These older
# browser runs were superseded and have no links in those documents.
SUPERSEDED_BROWSER_RUNS = ("20261105_153403", "20261105_153441")


def collect_files(root: Path) -> list[Path]:
    """Use a file-level allowlist independent of Git and local build outputs."""
    files = {root / name for name in REQUIRED_FILES}
    missing = sorted(path.relative_to(root).as_posix() for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing submission files: " + ", ".join(missing))

    for pattern in INCLUDED_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.is_relative_to(root / "docs") and path.stem.endswith(SUPERSEDED_BROWSER_RUNS):
                continue
            files.add(path)
    for path in files:
        # Reject filesystem aliases, including symlinks and junctions outside the source tree.
        if path.resolve(strict=True) != path or path.is_symlink():
            raise ValueError(f"Submission file must be a regular source path: {path.relative_to(root)}")

    manifest = json.loads((root / "data/samples.json").read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("The sample manifest has no demo inputs.")
    for item in manifest:
        for key in ("file", "ground_truth"):
            value = item[key]
            if key == "ground_truth" and value is None:
                continue
            relative = PurePosixPath(value)
            if (relative.is_absolute() or PureWindowsPath(value).drive or "\\" in value
                    or ".." in relative.parts or root / "data" / relative not in files):
                raise ValueError(f"Sample asset is not in the submission: {value}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def package_submission(root: Path, output: Path) -> int:
    """Build a deterministic archive and create the destination exclusively."""
    root = root.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("The output filename must end in .zip.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Archive already exists: {output}")
    files = collect_files(root)
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            entry = ZipInfo(f"{ARCHIVE_ROOT}/{relative}")
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            content = path.read_bytes()
            if relative.startswith("docs/results/") and path.suffix == ".xml":
                # Preserve linked test evidence without publishing the developer machine name.
                report = ElementTree.fromstring(content)
                for suite in report.iter("testsuite"):
                    suite.attrib.pop("hostname", None)
                content = ElementTree.tostring(report, encoding="utf-8", xml_declaration=True)
            archive.writestr(entry, content, compress_type=ZIP_DEFLATED, compresslevel=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload.seek(0)
    stream = output.open("xb")
    try:
        with stream:
            shutil.copyfileobj(payload, stream)
    except (OSError, KeyboardInterrupt):
        output.unlink(missing_ok=True)
        raise
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Path to a new .zip file.")
    args = parser.parse_args()
    try:
        count = package_submission(ROOT, args.output)
    except (OSError, ValueError, ElementTree.ParseError) as exc:
        parser.error(str(exc))
    print(f"{count} files · {args.output} · {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
