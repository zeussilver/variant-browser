from variant_browser.matching import (
    Overrides,
    load_overrides,
    normalise_id,
    reconcile,
    resolved_samples,
)

MAF_ROW = "TP53\t17\t100\tMissense_Mutation\tC\tT\t{sample}\t{normal}\tp.R273H\t900\t100\n"


def codes(issues, level=None):
    rows = issues if level is None else issues[issues["level"] == level]
    return set(zip(rows["code"], rows["id"]))


def test_a_clean_study_has_no_errors(write_study, config):
    study = write_study()
    report, issues = reconcile(study, config)
    assert codes(issues, "error") == set()
    assert (report["resolved_by"] == "sample table").all()
    assert len(resolved_samples(study, report)) == 4


def test_sample_table_disagrees_with_the_sample_id(write_study, config):
    study = write_study(
        append={"data_clinical_sample.txt": "P-02\tP-01_cfDNA_3\t300\tBLOOD\t0.2\n"}
    )
    _, issues = reconcile(study, config)
    assert ("PATIENT_CONFLICT", "P-01_cfDNA_3") in codes(issues, "error")


def test_tumour_called_against_another_patients_normal(write_study, config):
    study = write_study(
        append={
            "data_clinical_sample.txt": "P-02\tP-02_cfDNA_2\t90\tBLOOD\t0.1\n",
            "data_mutations.txt": MAF_ROW.format(sample="P-02_cfDNA_2", normal="P-01_WBC"),
        }
    )
    _, issues = reconcile(study, config)
    assert ("NORMAL_MISMATCH", "P-02_cfDNA_2") in codes(issues, "error")


def test_sequenced_sample_missing_from_the_table_is_reported_not_adopted(write_study, config):
    study = write_study(
        append={"data_mutations.txt": MAF_ROW.format(sample="P-02_cfDNA_9", normal="P-02_WBC")}
    )
    report, issues = reconcile(study, config)
    assert ("SAMPLE_NOT_IN_TABLE", "P-02_cfDNA_9") in codes(issues, "error")
    row = report.set_index("sample_id").loc["P-02_cfDNA_9"]
    assert row["resolved_by"] == "unresolved" and row["proposed_patient"] == "P-02"
    # A proposal is not a decision: the views must not show this sample yet.
    assert "P-02_cfDNA_9" not in set(resolved_samples(study, report)["sample_id"])


def test_same_id_typed_differently_is_flagged_and_fixed_only_by_a_person(write_study, config):
    study = write_study(
        append={"data_mutations.txt": MAF_ROW.format(sample="p01 cfdna-2", normal="P-01_WBC")}
    )
    report, issues = reconcile(study, config)
    assert ("NEAR_DUPLICATE_ID", "p01 cfdna-2") in codes(issues, "error")
    assert normalise_id("p01 cfdna-2") == normalise_id("P-01_cfDNA_2")

    decided = Overrides(same_as={"p01 cfdna-2": "P-01_cfDNA_2"})
    report, issues = reconcile(study, config, decided)
    assert codes(issues, "error") == set()
    assert ("ALIAS_APPLIED", "p01 cfdna-2") in codes(issues, "info")
    assert "p01 cfdna-2" not in set(report["sample_id"])


def test_override_sets_the_patient_and_is_recorded(write_study, config):
    study = write_study(
        append={"data_mutations.txt": MAF_ROW.format(sample="P-02_cfDNA_9", normal="P-02_WBC")}
    )
    report, issues = reconcile(study, config, Overrides(patient_of={"P-02_cfDNA_9": "P-02"}))
    row = report.set_index("sample_id").loc["P-02_cfDNA_9"]
    assert (row["patient_id"], row["resolved_by"]) == ("P-02", "override")
    assert ("OVERRIDE_APPLIED", "P-02_cfDNA_9") in codes(issues, "info")
    assert codes(issues, "error") == set()


def test_specimen_timeline_disagreements(write_study, config):
    study = write_study(
        append={
            "data_timeline_sample.txt": (
                "P-02\t200\t\tSPECIMEN\tP-01_cfDNA_2\tBLOOD\tBLOOD\n"  # other patient
                "P-02\t75\t\tSPECIMEN\tP-02_cfDNA_1\tBLOOD\tBLOOD\n"  # table says day 50
            )
        }
    )
    report, issues = reconcile(study, config)
    assert ("TIMELINE_PATIENT_CONFLICT", "P-01_cfDNA_2") in codes(issues, "error")
    assert ("DAY_CONFLICT", "P-02_cfDNA_1") in codes(issues, "warning")
    assert report.set_index("sample_id").loc["P-02_cfDNA_1", "collection_day"] == 50


def test_timeline_fills_in_a_missing_collection_day(write_study, config):
    study = write_study(
        append={
            "data_clinical_sample.txt": "P-02\tP-02_cfDNA_2\t\tBLOOD\t\nP-02\tP-02_cfDNA_3\t\tBLOOD\t\n",
            "data_timeline_sample.txt": "P-02\t120\t\tSPECIMEN\tP-02_cfDNA_2\tBLOOD\tBLOOD\n",
        }
    )
    report, issues = reconcile(study, config)
    days = report.set_index("sample_id")["collection_day"]
    assert days["P-02_cfDNA_2"] == 120
    assert ("NO_COLLECTION_DAY", "P-02_cfDNA_3") in codes(issues)
    assert ("NO_COLLECTION_DAY", "P-02_cfDNA_2") not in codes(issues)


def test_patient_and_row_level_checks(write_study, config):
    study = write_study(
        append={
            "data_timeline_treatment.txt": (
                "P-99\t0\t10\tTREATMENT\tMedical Therapy\t\tDOCETAXEL\tMETASTATIC\n"
                "P-02\t80\t60\tTREATMENT\tMedical Therapy\t\tDocetaxel\tMETASTATIC\n"
            ),
            "data_timeline_psa.txt": "P-02\t10\t\tLAB_TEST\tPSA\t2-3\n",
            "data_clinical_sample.txt": "P-77\tP-77_cfDNA_1\t5\tBLOOD\t\n",
        }
    )
    _, issues = reconcile(study, config)
    found = codes(issues)
    assert ("PATIENT_NOT_IN_TABLE", "P-99") in found
    assert ("STOP_BEFORE_START", "P-02") in found
    assert ("LAB_UNPARSEABLE", "P-02") in found
    assert ("AGENT_SPELLING", "DOCETAXEL") in found
    assert ("UNKNOWN_PATIENT", "P-77_cfDNA_1") in found
    assert ("NO_MUTATION_CALLS", "P-77_cfDNA_1") in found


def test_overrides_file(tmp_path):
    path = tmp_path / "overrides.tsv"
    path.write_text(
        "sample_id\tsame_as\tpatient_id\tnote\n"
        "a-1\tA_1\t\tretyped\n"
        "B_2\t\tP-02\tlab book\n"
    )
    loaded = load_overrides(path)
    assert loaded == Overrides(patient_of={"B_2": "P-02"}, same_as={"a-1": "A_1"})
    assert load_overrides(None) == Overrides()
