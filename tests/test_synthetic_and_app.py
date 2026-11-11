import importlib.util
from pathlib import Path

import pytest

from variant_browser.app import DEFAULT_STUDY, create_app
from variant_browser.io import load_config, load_study
from variant_browser.matching import load_overrides, reconcile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_demo.py"


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_synthetic_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("synthetic")
    module.main(out)
    other = tmp_path_factory.mktemp("synthetic_again")
    module.main(other)
    for name in ("data_mutations.txt", "data_clinical_sample.txt", "data_timeline_psa.txt"):
        assert (out / name).read_text() == (other / name).read_text(), f"{name} is not reproducible"
    return out


def problems(issues):
    rows = issues[issues["level"].isin(["error", "warning"])]
    return set(zip(rows["code"], rows["id"]))


def test_every_planted_problem_is_found(synthetic_dir):
    config = load_config()
    study = load_study(synthetic_dir, config)
    assert study.is_synthetic
    _, issues = reconcile(study, config)
    assert problems(issues) >= {
        ("NEAR_DUPLICATE_ID", "syn003-cfdna-c3"),
        ("PATIENT_CONFLICT", "SYN-004_cfDNA_C2"),
        ("NORMAL_MISMATCH", "SYN-006_cfDNA_C4"),
        ("SAMPLE_NOT_IN_TABLE", "SYN-001_cfDNA_C5"),
        ("STOP_BEFORE_START", "SYN-005"),
    }
    assert ("AGENT_SPELLING", "LEUPROLIDE") in set(zip(issues["code"], issues["id"]))


def test_recorded_decisions_clear_exactly_the_problems_they_address(synthetic_dir):
    config = load_config()
    study = load_study(synthetic_dir, config)
    decisions = load_overrides(synthetic_dir / "overrides.example.tsv")
    report, issues = reconcile(study, config, decisions)
    # Nobody has decided about the swapped normal or the backwards dates.
    assert problems(issues) == {
        ("NORMAL_MISMATCH", "SYN-006_cfDNA_C4"),
        ("STOP_BEFORE_START", "SYN-005"),
    }
    patients = report.set_index("sample_id")["patient_id"]
    assert patients["SYN-004_cfDNA_C2"] == "SYN-004"
    assert patients["SYN-001_cfDNA_C5"] == "SYN-001"
    assert "syn003-cfdna-c3" not in patients.index


def test_app_builds_on_the_synthetic_study(synthetic_dir):
    config = load_config()
    study = load_study(synthetic_dir, config)
    app = create_app(study, config, load_overrides(synthetic_dir / "overrides.example.tsv"))
    text = str(app.layout)
    assert "synthetic demonstration data" in text
    assert "SYN-001" in text


@pytest.mark.skipif(not DEFAULT_STUDY.exists(), reason="public cohort not downloaded")
def test_public_cohort_loads_with_the_expected_shape():
    config = load_config()
    study = load_study(DEFAULT_STUDY, config)
    assert not study.is_synthetic
    assert (len(study.patients), len(study.samples)) == (97, 123)
    assert study.mutations["sample_id"].nunique() == 82
    report, issues = reconcile(study, config)
    counts = issues.groupby("code").size()
    assert counts["PATIENT_NOT_IN_TABLE"] == 62
    assert counts["SAMPLE_NOT_IN_TABLE"] == 5
    assert counts["DAY_UNPARSEABLE"] == 1  # the '#VALUE!' stop day
    assert not issues["level"].eq("error").any()
    create_app(study, config)
