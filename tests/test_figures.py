import pytest

from variant_browser import figures, variants
from variant_browser.matching import reconcile, resolved_samples


@pytest.fixture
def loaded(write_study, config):
    study = write_study()
    report, _ = reconcile(study, config)
    samples = resolved_samples(study, report)
    return study, samples, variants.protein_altering(study.mutations, samples)


def timeline(loaded, config, patient="P-01", **kwargs):
    study, samples, calls = loaded
    pick = lambda frame: frame[frame["patient_id"] == patient]  # noqa: E731
    return figures.patient_timeline(
        patient, pick(samples), pick(calls), pick(study.treatments), pick(study.labs),
        pick(study.status_events), config.gene_panel, **kwargs,
    )  # fmt: skip


def titles(fig):
    return [a.text for a in fig.layout.annotations]


def test_heatmap_cells(loaded):
    _, samples, calls = loaded
    genes, ids = ["TP53", "AR", "SPOP"], samples["sample_id"].tolist()
    grid = variants.gene_sample_grid(calls, genes, ids)
    gene_order, sample_order = variants.order_for_heatmap(grid, genes, ids)
    fig = figures.cohort_heatmap(grid, gene_order, sample_order, samples)
    cells = fig.data[1]
    assert [label.split()[0] for label in cells.y] == ["TP53", "AR", "SPOP"]
    assert cells.y[0].endswith("75%")  # TP53 in 3 of 4 samples
    # First column is P-01_cfDNA_2: truncating TP53 (code 3) with two variants, missense AR (1).
    assert [row[0] for row in cells.z] == [3, 1, 0]
    assert cells.text[0][0] == "2" and cells.text[1][0] == ""
    assert {t.name for t in fig.data if t.name} >= {"Truncating", "Missense", "In-frame indel"}


def test_heatmap_with_nothing_to_show_says_so():
    import pandas as pd

    fig = figures.cohort_heatmap(pd.DataFrame(), [], [], pd.DataFrame())
    assert "No samples" in fig.layout.annotations[0].text


def test_timeline_has_one_panel_per_kind_of_data(loaded, config):
    fig = timeline(loaded, config)
    assert titles(fig)[:4] == [
        "Treatments", "Variant allele fraction in each sequenced sample",
        "ctDNA tumour fraction", "PSA (log scale)",
    ]  # fmt: skip
    # Each measure has its own y-axis; none is overlaid on another.
    assert all(getattr(trace, "yaxis", None) != "y5" for trace in fig.data)
    assert not any(axis.overlaying for axis in fig.select_yaxes())


def test_timeline_leaves_out_panels_with_no_data(loaded, config):
    fig = timeline(loaded, config, patient="P-02")  # no PSA
    assert "PSA (log scale)" not in titles(fig)
    assert "Treatments" in titles(fig)


def test_lines_join_plasma_samples_only(loaded, config):
    fig = timeline(loaded, config)
    lines = [t for t in fig.data if t.type == "scatter" and t.mode == "lines" and t.legendgroup == "Missense"]
    assert len(lines) == 1
    # TP53 p.R273H is in tissue (day 0) and plasma (days 100, 200); only plasma is joined.
    assert list(lines[0].x) == [100, 200, None]


def test_open_ended_treatment_runs_to_the_last_record_and_is_hatched(loaded, config):
    fig = timeline(loaded, config)
    hatched = [t for t in fig.data if t.type == "bar" and t.marker.pattern.shape == "/" and t.x[0] is not None]
    assert len(hatched) == 1 and list(hatched[0].y) == ["Leuprolide"]
    assert hatched[0].base[0] + hatched[0].x[0] == 200  # last PSA and last sample


def test_result_below_detection_limit_is_drawn_open(loaded, config):
    fig = timeline(loaded, config)
    psa = next(t for t in fig.data if t.name == "PSA")
    assert list(psa.y) == [42.0, 0.05, 7.5]
    assert psa.marker.color[1] == figures.SURFACE and psa.marker.color[0] != figures.SURFACE


def test_each_variant_is_labelled_once_at_its_last_sample(loaded, config):
    fig = timeline(loaded, config)
    labels = [a.text for a in fig.layout.annotations if a.showarrow]
    assert sorted(labels) == ["AR p.T878A", "TP53 p.Q331*", "TP53 p.R273H"]


def test_variants_outside_the_panel_are_opt_in(loaded, config):
    study, samples, calls = loaded
    names = lambda fig: {t.name for t in fig.data}  # noqa: E731
    pick = lambda frame: frame[frame["patient_id"] == "P-01"]  # noqa: E731
    args = (pick(samples), pick(calls), pick(study.treatments), pick(study.labs), pick(study.status_events))
    ar_only = ("AR",)  # TP53 is now outside the panel
    assert "Variants outside the gene panel" not in names(figures.patient_timeline("P-01", *args, ar_only))
    assert "Variants outside the gene panel" in names(
        figures.patient_timeline("P-01", *args, ar_only, show_other_variants=True)
    )


@pytest.mark.parametrize(
    "values", [[0.30, 0.29, 0.29, 0.28], [0.5, 0.1], [0.2], [0.05, 0.05, 0.05, 0.05, 0.05]]
)
def test_spread_labels_keeps_the_gap_and_the_centre(values):
    gap = 0.03
    placed = figures.spread_labels(values, gap)
    assert all(a - b >= gap - 1e-9 for a, b in zip(placed, placed[1:]))
    assert sum(placed) == pytest.approx(sum(values))


def test_short_sample_labels(loaded):
    _, samples, _ = loaded
    labels = figures.short_sample_labels(samples)
    assert labels == {"P-01_T1": "1 T", "P-01_cfDNA_1": "1 B", "P-01_cfDNA_2": "1 B2", "P-02_cfDNA_1": "2 B"}
