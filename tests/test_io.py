import pandas as pd
import pytest

from variant_browser.io import normalise_agent, parse_lab_result


def test_description_rows_are_skipped_but_a_hash_inside_a_field_is_kept(write_study):
    # The public cohort has a stop day of '#VALUE!'. Reading with pandas'
    # comment='#' cuts that row off after the start day and loses the drug.
    study = write_study(
        replace={
            "data_clinical_patient.txt": "#Patient Identifier\n#STRING\nPATIENT_ID\nP-01\nP-02\n"
        },
        append={
            "data_timeline_treatment.txt": "P-02\t60\t#VALUE!\tTREATMENT\tMedical Therapy\t\tBICALUTAMIDE\tMETASTATIC\n"
        },
    )
    assert study.patients["patient_id"].tolist() == ["P-01", "P-02"]
    row = study.treatments[study.treatments["agent"] == "BICALUTAMIDE"].iloc[0]
    assert row["start_day"] == 60 and pd.isna(row["stop_day"])
    assert ("DAY_UNPARSEABLE", "P-02") == study.load_notes[0][:2]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5.55", (5.55, "")),
        ("0", (0.0, "")),
        ("<0.1", (0.1, "<")),
        ("<0. 1", (0.1, "<")),
        (">1000", (1000.0, ">")),
        ("2-3", (None, "unparseable")),
        ("", (None, "unparseable")),
        (None, (None, "unparseable")),
    ],
)
def test_lab_results(text, expected):
    assert parse_lab_result(text) == expected


def test_drug_spellings_collapse_to_one_name():
    assert {normalise_agent(n) for n in ["Leuprolide", "LEUPROLIDE ", " leuprolide"]} == {"LEUPROLIDE"}
    assert normalise_agent(None) == ""


def test_allele_fraction_is_unknown_not_zero_when_there_are_no_reads(write_study):
    study = write_study(
        append={"data_mutations.txt": "PTEN\t10\t600\tNonsense_Mutation\tC\tT\tP-02_cfDNA_1\tP-02_WBC\tp.R130*\t0\t0\n"}
    )
    pten = study.mutations[study.mutations["gene"] == "PTEN"].iloc[0]
    assert pd.isna(pten["vaf"])
    tp53 = study.mutations[study.mutations["sample_id"] == "P-01_T1"].iloc[0]
    assert tp53["vaf"] == pytest.approx(0.2)


def test_sample_type_and_treatment_class(write_study):
    study = write_study(
        append={"data_timeline_treatment.txt": "P-02\t5\t9\tTREATMENT\tMedical Therapy\t\tASPIRIN\tOTHER\n"}
    )
    types = dict(zip(study.samples["sample_id"], study.samples["sample_type"]))
    assert types == {"P-01_T1": "Tissue", "P-01_cfDNA_1": "Blood", "P-01_cfDNA_2": "Blood", "P-02_cfDNA_1": "Blood"}
    classes = dict(zip(study.treatments["agent"], study.treatments["treatment_class"]))
    assert classes["ENZALUTAMIDE"] == "AR pathway inhibitor"
    assert classes["ASPIRIN"] == "Supportive / other"  # not in the class table
