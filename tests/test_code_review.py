"""Regression contracts raised by CR-01, CR-02 and CR-10."""
import numpy as np
import pytest

from src.algorithms.kmeans import kmeans, unique_rows
from src.processing.pipeline import KMeansParams, default_foreground, kmeans_outputs, prepare_input, run_kmeans, run_otsu
from src.utils.image_io import load_image_bytes, to_png_bytes
from src.utils.metrics import compare_masks, compare_trimap


@pytest.mark.parametrize("max_iter,tol", [(1, 0.0), (2, 0.0), (100, 0.0), (100, 50.0)])
@pytest.mark.parametrize("init", ["random", "kmeans++"])
def test_history_is_sse_after_each_complete_lloyd_step(max_iter, tol, init):
    data = np.random.default_rng(83).normal(size=(130, 3)) * 20
    weights = np.random.default_rng(12).integers(1, 8, len(data))
    result = kmeans(data, 7, weights=weights, init=init, max_iter=max_iter, tol=tol, seed=6)
    expected = []
    for centers in result.centers_history:
        squared = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        expected.append(float(np.dot(weights, squared)))
    assert result.initial_sse == pytest.approx(expected[0])
    np.testing.assert_allclose(result.sse_history, expected[1:], rtol=1e-12)
    assert result.sse_history[-1] == result.sse
    assert len(result.sse_history) == result.n_iter


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_byte_unique_preserves_sorted_vectors_across_validation_chunks(channels):
    data = np.random.default_rng(97).integers(0, 256, (270000, channels)).astype(np.float64)
    data[-1] = data[0]
    np.testing.assert_array_equal(unique_rows(data), np.unique(data, axis=0))
    data[-1, 0] = 12.25
    np.testing.assert_array_equal(unique_rows(data), np.unique(data, axis=0))


def test_border_object_documents_failure_and_manual_cluster_selection_recovers():
    image = np.full((80, 80, 3), 220, dtype=np.uint8)
    image[:, :12] = 20  # Object touches three edges and misses the center.
    gt = np.zeros((80, 80), dtype=np.uint8)
    gt[:, :12] = 255
    loaded = load_image_bytes(to_png_bytes(image), "edge.png")
    prepared = prepare_input(loaded, 800)
    result = run_kmeans(prepared, KMeansParams(k=2))
    automatic = kmeans_outputs(prepared, result, default_foreground(result))
    manual = kmeans_outputs(prepared, result, [0])
    assert compare_masks(automatic.mask, gt).iou < 0.1  # Heuristic has a documented limit.
    assert compare_masks(manual.mask, gt).iou == 1.0
    assert compare_masks(run_otsu(prepared, polarity="dark").mask, gt).iou == 1.0


@pytest.mark.parametrize("gain", [0.35, 0.7, 1.0])
def test_low_light_separated_object_keeps_scores(gain):
    image = np.full((60, 70), int(220 * gain), dtype=np.uint8)
    image[15:45, 20:50] = int(40 * gain)
    gt = np.zeros_like(image)
    gt[15:45, 20:50] = 255
    prepared = prepare_input(load_image_bytes(to_png_bytes(image), "brightness.png"), 800)
    km = run_kmeans(prepared, KMeansParams(k=2, feature="gray"))
    assert compare_masks(kmeans_outputs(prepared, km, default_foreground(km)).mask, gt).iou == 1.0
    assert compare_masks(run_otsu(prepared, polarity="auto").mask, gt).iou == 1.0


def test_trimap_uncertain_pixels_do_not_affect_scores():
    truth = np.array([[0, 128, 255], [0, 128, 255]], dtype=np.uint8)
    for uncertain in (0, 255):
        pred = np.array([[0, uncertain, 255], [0, uncertain, 255]], dtype=np.uint8)
        score = compare_trimap(pred, truth)
        assert score.iou == score.dice == 1.0
        assert score.tp + score.tn == 4


@pytest.mark.parametrize("truth", [np.full((2, 3), 128), np.full((2, 3), 64)])
def test_trimap_rejects_no_known_pixels_or_unknown_labels(truth):
    with pytest.raises(ValueError):
        compare_trimap(np.zeros((2, 3)), truth)


def test_lockfiles_pin_every_package_and_separate_development_dependencies():
    from pathlib import Path
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    root = Path(__file__).resolve().parents[1]
    def packages(name):
        result = {}
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            req = Requirement(line)
            assert len(list(req.specifier)) == 1
            assert next(iter(req.specifier)).operator == "=="
            result[canonicalize_name(req.name)] = str(req.specifier)
        return result
    runtime, development = packages("requirements.txt"), packages("requirements-dev.txt")
    assert not {"pytest", "ruff", "pip-tools"}.intersection(runtime)
    assert {"pytest", "ruff", "pip-tools"}.issubset(development)
    assert all(development[name] == version for name, version in runtime.items())


def test_invert_foreground_updates_downloaded_mask_and_report():
    import io
    import json
    from pathlib import Path
    from PIL import Image
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60)
    app.session_state["_test_page"] = "kmeans"
    app.run()
    next(button for button in app.button if button.label == "Chạy K-means").click().run()
    assert not app.exception
    before_files = app.session_state["km_result"]["_export_cache"]["files"]
    before = np.asarray(Image.open(io.BytesIO(before_files["kmeans_mask.png"][0])))
    before_clusters = json.loads(before_files["kmeans_stats.json"][0])["cum_doi_tuong"]
    app.checkbox(key="km_invert_foreground").set_value(True).run()
    assert not app.exception
    after_files = app.session_state["km_result"]["_export_cache"]["files"]
    after = np.asarray(Image.open(io.BytesIO(after_files["kmeans_mask.png"][0])))
    np.testing.assert_array_equal(after, 255 - before)
    after_clusters = json.loads(after_files["kmeans_stats.json"][0])["cum_doi_tuong"]
    assert not set(before_clusters).intersection(after_clusters)
