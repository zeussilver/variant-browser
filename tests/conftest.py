"""A hand-written study small enough to check by eye.

Two patients. P-01 has a tissue sample and two plasma samples; P-02 has one
plasma sample. Each test that needs a problem adds it on top of this.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from variant_browser.io import Config, IdRule, load_study

PATIENTS = "PATIENT_ID\nP-01\nP-02\n"
SAMPLES = (
    "PATIENT_ID\tSAMPLE_ID\tBX_TIME_DAYS\tBX_LOCATION\tTUMOR_FRACTION\n"
    "P-01\tP-01_T1\t0\tPROSTATE\t\n"
    "P-01\tP-01_cfDNA_1\t100\tBLOOD\t0.30\n"
    "P-01\tP-01_cfDNA_2\t200\tBLOOD\t0.10\n"
    "P-02\tP-02_cfDNA_1\t50\tBLOOD\t0.05\n"
)
MAF_HEADER = (
    "Hugo_Symbol\tChromosome\tStart_Position\tVariant_Classification\tReference_Allele\t"
    "Tumor_Seq_Allele2\tTumor_Sample_Barcode\tMatched_Norm_Sample_Barcode\tHGVSp_Short\t"
    "t_ref_count\tt_alt_count\n"
)
MUTATIONS = MAF_HEADER + (
    "TP53\t17\t100\tMissense_Mutation\tC\tT\tP-01_T1\tP-01_WBC\tp.R273H\t80\t20\n"
    "TP53\t17\t100\tMissense_Mutation\tC\tT\tP-01_cfDNA_1\tP-01_WBC\tp.R273H\t850\t150\n"
    "TP53\t17\t100\tMissense_Mutation\tC\tT\tP-01_cfDNA_2\tP-01_WBC\tp.R273H\t950\t50\n"
    "TP53\t17\t200\tNonsense_Mutation\tG\tA\tP-01_cfDNA_2\tP-01_WBC\tp.Q331*\t980\t20\n"
    "AR\tX\t300\tMissense_Mutation\tA\tG\tP-01_cfDNA_2\tP-01_WBC\tp.T878A\t990\t10\n"
    "TTN\t2\t400\tSilent\tG\tA\tP-01_cfDNA_1\tP-01_WBC\tp.A10A\t900\t100\n"
    "SPOP\t17\t500\tIn_Frame_Del\tAGG\tA\tP-02_cfDNA_1\tP-02_WBC\tp.F133del\t970\t30\n"
)
TREATMENTS = (
    "PATIENT_ID\tSTART_DATE\tSTOP_DATE\tEVENT_TYPE\tTREATMENT_TYPE\tSUBTYPE\tAGENT\tCOMMENTS\n"
    "P-01\t0\t\tTREATMENT\tMedical Therapy\t\tLEUPROLIDE\tMETASTATIC\n"
    "P-01\t10\t150\tTREATMENT\tMedical Therapy\t\tENZALUTAMIDE\tMETASTATIC\n"
    "P-02\t0\t40\tTREATMENT\tMedical Therapy\t\tDOCETAXEL\tMETASTATIC\n"
)
PSA = (
    "PATIENT_ID\tSTART_DATE\tSTOP_DATE\tEVENT_TYPE\tTEST\tRESULT\n"
    "P-01\t0\t\tLAB_TEST\tPSA\t42.0\n"
    "P-01\t100\t\tLAB_TEST\tPSA\t<0.05\n"
    "P-01\t200\t\tLAB_TEST\tPSA\t7.5\n"
)
SPECIMENS = (
    "PATIENT_ID\tSTART_DATE\tSTOP_DATE\tEVENT_TYPE\tSAMPLE_ID\tSPECIMEN_TYPE\tSPECIMEN_SITE\n"
    "P-01\t0\t\tSPECIMEN\tP-01_T1\tTISSUE\tPROSTATE\n"
    "P-01\t100\t\tSPECIMEN\tP-01_cfDNA_1\tBLOOD\tBLOOD\n"
)

FILES = {
    "data_clinical_patient.txt": PATIENTS,
    "data_clinical_sample.txt": SAMPLES,
    "data_mutations.txt": MUTATIONS,
    "data_timeline_treatment.txt": TREATMENTS,
    "data_timeline_psa.txt": PSA,
    "data_timeline_sample.txt": SPECIMENS,
}


@pytest.fixture
def config() -> Config:
    return Config(
        time_axis_label="Day",
        id_rules=(IdRule(re.compile(r"^(?P<pt>P-\d{2})"), "{pt}"),),
        gene_panel=("TP53", "AR", "SPOP"),
        treatment_classes={
            "LEUPROLIDE": "Hormonal",
            "ENZALUTAMIDE": "AR pathway inhibitor",
            "DOCETAXEL": "Chemotherapy",
        },
    )


@pytest.fixture
def write_study(tmp_path: Path, config: Config):
    """Write the small study, with some files replaced or appended to, and load it."""

    def _write(replace: dict[str, str] | None = None, append: dict[str, str] | None = None):
        files = {**FILES, **(replace or {})}
        for name, extra in (append or {}).items():
            files[name] = files[name] + extra
        for name, text in files.items():
            (tmp_path / name).write_text(text)
        return load_study(tmp_path, config)

    return _write
