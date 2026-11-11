import pytest

from variant_browser import variants
from variant_browser.matching import reconcile, resolved_samples


@pytest.fixture
def calls(write_study, config):
    study = write_study()
    report, _ = reconcile(study, config)
    return variants.protein_altering(study.mutations, resolved_samples(study, report))


def test_only_protein_altering_calls_are_kept(calls):
    assert "TTN" not in set(calls["gene"])  # the silent call
    assert set(calls["effect"]) == {"Missense", "Truncating", "In-frame indel"}
    assert calls.loc[calls["gene"] == "SPOP", "patient_id"].item() == "P-02"


def test_filters(write_study, config):
    study = write_study()
    report, _ = reconcile(study, config)
    samples = resolved_samples(study, report)
    assert len(variants.protein_altering(study.mutations, samples, min_vaf=0.1)) == 2
    assert len(variants.protein_altering(study.mutations, samples, min_alt_reads=25)) == 3


def test_a_cell_takes_its_most_damaging_variant_and_counts_them(calls):
    samples = ["P-01_T1", "P-01_cfDNA_1", "P-01_cfDNA_2", "P-02_cfDNA_1"]
    grid = variants.gene_sample_grid(calls, ["TP53", "AR", "SPOP"], samples)
    cell = grid.set_index(["gene", "sample_id"]).loc[("TP53", "P-01_cfDNA_2")]
    assert cell["effect"] == "Truncating" and cell["n_variants"] == 2
    assert cell["max_vaf"] == pytest.approx(0.05)
    assert len(grid) == 5


def test_heatmap_order(calls):
    genes, samples = ["AR", "SPOP", "TP53"], ["P-02_cfDNA_1", "P-01_T1", "P-01_cfDNA_2", "P-01_cfDNA_1"]
    grid = variants.gene_sample_grid(calls, genes, samples)
    gene_order, sample_order = variants.order_for_heatmap(grid, genes, samples)
    assert gene_order == ["TP53", "AR", "SPOP"]  # most samples first, then the order given
    # TP53 and AR first, then TP53 only, then the sample with neither.
    assert sample_order == ["P-01_cfDNA_2", "P-01_T1", "P-01_cfDNA_1", "P-02_cfDNA_1"]


def test_most_altered_genes_counts_patients_not_calls(calls):
    # TP53 has four calls but they are all in one patient, so it does not
    # outrank AR and SPOP. Ties go alphabetically.
    assert variants.most_altered_genes(calls, 3) == ["AR", "SPOP", "TP53"]


def test_serial_variants(calls):
    of_patient = calls[calls["patient_id"] == "P-01"]
    tracked = variants.serial_variant_keys(of_patient)
    assert tracked == {"17:100:C:T"}  # TP53 p.R273H, in all three P-01 samples
