"""Write a small synthetic study with serial plasma samples.

The public cohort has almost no patient with more than one sequenced sample,
so it cannot show the main use of the timeline view: following the same
variants through treatment. This script makes up six patients who each have a
tissue sample and three to five plasma samples.

NOTHING HERE IS REAL. Patients, dates, allele fractions and PSA values are
invented with a fixed random seed. The gene names and protein changes are real
ones so the labels look familiar; their behaviour here is made up.

The study also contains deliberate ID problems (listed in SYNTHETIC.txt) so
that the data checks page has something to find.

    python scripts/make_synthetic_demo.py            # writes data/synthetic_serial
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_serial"
SEED = 20261111

# gene, protein change, class, chrom, pos, ref, alt
TRUNCAL = [
    ("TP53", "p.R273H", "Missense_Mutation", "17", 7577120, "C", "T"),
    ("PTEN", "p.R130*", "Nonsense_Mutation", "10", 89692904, "C", "T"),
    ("SPOP", "p.F133L", "Missense_Mutation", "17", 47696425, "A", "G"),
    ("BRCA2", "p.S1982Rfs*22", "Frame_Shift_Del", "13", 32914437, "GT", "G"),
    ("CDK12", "p.E928Nfs*10", "Frame_Shift_Del", "17", 37657600, "AG", "A"),
    ("FOXA1", "p.M253_N256del", "In_Frame_Del", "14", 38061200, "CATGTTGTTC", "C"),
]
RESISTANCE = [
    ("AR", "p.T878A", "Missense_Mutation", "X", 66943552, "A", "G"),
    ("AR", "p.L702H", "Missense_Mutation", "X", 66931463, "T", "A"),
    ("RB1", "p.R552*", "Nonsense_Mutation", "13", 48955538, "C", "T"),
]
PASSENGER = [
    ("TTN", "p.A1021V", "Missense_Mutation", "2", 179640000, "G", "A"),
    ("MUC16", "p.S3045L", "Missense_Mutation", "19", 9060000, "G", "A"),
    ("OBSCN", "p.R4831Q", "Missense_Mutation", "1", 228500000, "G", "A"),
    ("RYR2", "p.E1127K", "Missense_Mutation", "1", 237700000, "G", "A"),
]

# Per patient: plasma days, tumour fraction at each, which truncal variants,
# which resistance variant appears from which plasma sample, and treatments.
PATIENTS = {
    "SYN-001": dict(
        plasma=[30, 210, 420, 610, 760], fraction=[0.42, 0.06, 0.03, 0.21, 0.48],
        truncal=[0, 1], resistance=(0, 3),
        treatments=[("LEUPROLIDE", 0, None), ("ENZALUTAMIDE", 45, 640), ("DOCETAXEL", 655, 770)],
    ),
    "SYN-002": dict(
        plasma=[10, 180, 365], fraction=[0.31, 0.02, 0.01],
        truncal=[2, 5], resistance=None,
        treatments=[("LEUPROLIDE", 0, None), ("ABIRATERONE", 20, None)],
    ),
    "SYN-003": dict(
        plasma=[60, 240, 400, 560], fraction=[0.55, 0.35, 0.12, 0.40],
        truncal=[0, 3], resistance=(2, 3),
        treatments=[("Leuprolide", 0, None), ("DOCETAXEL", 70, 190), ("OLAPARIB", 250, 540),
                    ("CABAZITAXEL", 575, 600)],
    ),
    "SYN-004": dict(
        plasma=[15, 150, 330, 500], fraction=[0.18, 0.04, 0.09, 0.27],
        truncal=[0, 4], resistance=(1, 2),
        treatments=[("LEUPROLIDE", 0, None), ("ABIRATERONE", 30, 345), ("ENZALUTAMIDE", 350, 510)],
    ),
    "SYN-005": dict(
        plasma=[90, 300, 480], fraction=[0.08, 0.02, 0.05],
        truncal=[1, 2], resistance=None,
        treatments=[("DEGARELIX", 0, None), ("ENZALUTAMIDE", 100, None), ("RADIUM-223", 470, 430)],
    ),
    "SYN-006": dict(
        plasma=[20, 200, 390, 540, 700], fraction=[0.61, 0.20, 0.05, 0.11, 0.37],
        truncal=[0, 1, 5], resistance=(0, 3),
        treatments=[("LEUPROLIDE", 0, None), ("DOCETAXEL", 25, 150), ("ABIRATERONE", 160, 560),
                    ("LUTETIUM-177", 580, 690)],
    ),
}


def main(out: Path = OUT) -> None:
    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    samples, calls, treatments, psa, specimens = [], [], [], [], []

    def call(sample_id: str, normal_id: str, variant: tuple, allele_fraction: float, depth: int) -> None:
        gene, change, cls, chrom, pos, ref, alt = variant
        alt_reads = int(rng.binomial(depth, min(max(allele_fraction, 0.0), 1.0)))
        if alt_reads < 3:  # below what a caller would report
            return
        calls.append(
            dict(Hugo_Symbol=gene, Chromosome=chrom, Start_Position=pos, Variant_Classification=cls,
                 Reference_Allele=ref, Tumor_Seq_Allele2=alt, Tumor_Sample_Barcode=sample_id,
                 Matched_Norm_Sample_Barcode=normal_id, HGVSp_Short=change,
                 t_ref_count=depth - alt_reads, t_alt_count=alt_reads)
        )  # fmt: skip

    for index, (patient, spec) in enumerate(PATIENTS.items()):
        normal_id = f"{patient}_WBC"
        truncal = [TRUNCAL[i] for i in spec["truncal"]]
        passengers = [PASSENGER[(index + k) % len(PASSENGER)] for k in range(2)]

        tissue_id = f"{patient}_FFPE_T1"
        samples.append(dict(PATIENT_ID=patient, SAMPLE_ID=tissue_id, BX_TIME_DAYS=0,
                            BX_LOCATION="PROSTATE", TUMOR_FRACTION=""))  # fmt: skip
        specimens.append(dict(PATIENT_ID=patient, START_DATE=0, STOP_DATE="", EVENT_TYPE="SPECIMEN",
                              SAMPLE_ID=tissue_id, SPECIMEN_TYPE="TISSUE", SPECIMEN_SITE="PROSTATE"))  # fmt: skip
        purity = rng.uniform(0.35, 0.7)
        for variant in truncal + passengers:
            call(tissue_id, normal_id, variant, purity / 2, int(rng.integers(120, 260)))

        for visit, (day, fraction) in enumerate(zip(spec["plasma"], spec["fraction"]), start=1):
            sample_id = f"{patient}_cfDNA_C{visit}"
            samples.append(dict(PATIENT_ID=patient, SAMPLE_ID=sample_id, BX_TIME_DAYS=day,
                                BX_LOCATION="BLOOD", TUMOR_FRACTION=round(fraction, 3)))  # fmt: skip
            specimens.append(dict(PATIENT_ID=patient, START_DATE=day, STOP_DATE="", EVENT_TYPE="SPECIMEN",
                                  SAMPLE_ID=sample_id, SPECIMEN_TYPE="BLOOD", SPECIMEN_SITE="BLOOD"))  # fmt: skip
            depth = int(rng.integers(900, 1600))
            for variant in truncal:
                call(sample_id, normal_id, variant, fraction / 2, depth)
            for variant in passengers:
                call(sample_id, normal_id, variant, fraction * 0.3, depth)
            if spec["resistance"]:
                which, first_visit = spec["resistance"]
                if visit >= first_visit:
                    share = 0.25 + 0.3 * (visit - first_visit)  # the resistant clone grows
                    call(sample_id, normal_id, RESISTANCE[which], fraction / 2 * min(share, 0.9), depth)
            # PSA loosely follows the tumour fraction, with two extra readings between visits.
            for offset in (-12, 0, 45):
                level = 220 * fraction * rng.lognormal(0, 0.25)
                result = "<0.05" if level < 0.6 else f"{level:.2f}"
                psa.append(dict(PATIENT_ID=patient, START_DATE=day + offset, STOP_DATE="",
                                EVENT_TYPE="LAB_TEST", TEST="PSA", RESULT=result))  # fmt: skip

        for agent, start, stop in spec["treatments"]:
            treatments.append(dict(PATIENT_ID=patient, START_DATE=start, STOP_DATE="" if stop is None else stop,
                                   EVENT_TYPE="TREATMENT", TREATMENT_TYPE="Medical Therapy", SUBTYPE="",
                                   AGENT=agent, COMMENTS="METASTATIC"))  # fmt: skip

    samples_frame, calls_frame = pd.DataFrame(samples), pd.DataFrame(calls)

    # --- Deliberate ID problems. Keep in step with SYNTHETIC.txt and the tests. ---
    # 1. The mutation file spells one sample differently from the sample table.
    calls_frame["Tumor_Sample_Barcode"] = calls_frame["Tumor_Sample_Barcode"].replace(
        {"SYN-003_cfDNA_C3": "syn003-cfdna-c3"}
    )
    # 2. The sample table gives a SYN-004 sample to SYN-005.
    wrong = samples_frame["SAMPLE_ID"] == "SYN-004_cfDNA_C2"
    samples_frame.loc[wrong, "PATIENT_ID"] = "SYN-005"
    # 3. One tumour sample was called against another patient's normal.
    swapped = calls_frame["Tumor_Sample_Barcode"] == "SYN-006_cfDNA_C4"
    calls_frame.loc[swapped, "Matched_Norm_Sample_Barcode"] = "SYN-002_WBC"
    # 4. A sequenced sample is missing from the sample table altogether.
    samples_frame = samples_frame[samples_frame["SAMPLE_ID"] != "SYN-001_cfDNA_C5"]

    def write(name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(out / name, sep="\t", index=False, lineterminator="\n")

    write("data_clinical_patient.txt", pd.DataFrame({"PATIENT_ID": list(PATIENTS)}))
    write("data_clinical_sample.txt", samples_frame)
    write("data_mutations.txt", calls_frame)
    write("data_timeline_treatment.txt", pd.DataFrame(treatments))
    write("data_timeline_psa.txt", pd.DataFrame(psa))
    write("data_timeline_sample.txt", pd.DataFrame(specimens))
    (out / "meta_study.txt").write_text(
        "type_of_cancer: prostate\nname: Synthetic serial ctDNA demo (not real patients)\n"
    )
    (out / "overrides.example.tsv").write_text(
        "sample_id\tsame_as\tpatient_id\tnote\n"
        "syn003-cfdna-c3\tSYN-003_cfDNA_C3\t\tSame tube, ID retyped by the sequencing provider\n"
        "SYN-004_cfDNA_C2\t\tSYN-004\tSample table row was pasted under the wrong patient\n"
        "SYN-001_cfDNA_C5\t\tSYN-001\tMissing from the sample table; confirmed against the lab book\n"
    )
    (out / "SYNTHETIC.txt").write_text(
        "This study is synthetic. Every patient, date and measurement was invented by\n"
        f"scripts/make_synthetic_demo.py (seed {SEED}). Do not use it for any analysis.\n\n"
        "Deliberate problems for the data checks page to find:\n"
        "1. NEAR_DUPLICATE_ID   'syn003-cfdna-c3' in the mutation file is 'SYN-003_cfDNA_C3' in the sample table.\n"
        "2. PATIENT_CONFLICT    'SYN-004_cfDNA_C2' is listed under SYN-005 in the sample table.\n"
        "3. NORMAL_MISMATCH     'SYN-006_cfDNA_C4' was called against the normal 'SYN-002_WBC'.\n"
        "4. SAMPLE_NOT_IN_TABLE 'SYN-001_cfDNA_C5' has variant calls but no row in the sample table.\n"
        "5. AGENT_SPELLING      'Leuprolide' and 'LEUPROLIDE'.\n"
        "6. STOP_BEFORE_START   SYN-005 radium-223 stops before it starts.\n\n"
        "overrides.example.tsv records a person's decision on problems 1, 2 and 4.\n"
    )
    print(f"Wrote {len(samples_frame)} samples and {len(calls_frame)} variant calls to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
