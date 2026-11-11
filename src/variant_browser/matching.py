"""Check that every sample is attached to the right patient.

The sample table says which patient each sample belongs to. This module does
not trust it blindly and does not quietly repair it either. It compares the
table against every other place a sample or patient ID appears:

* the patient ID written inside the sample ID itself (rules in config/study.toml)
* the mutation file: tumour sample and its matched normal
* the specimen timeline: patient and collection day
* the treatment and lab timelines: patient IDs

and reports each disagreement with a code. A person then decides. Decisions go
in an overrides file; the code never guesses. The file has four columns:

    sample_id   the ID as it appears in the data
    same_as     another sample ID that this one is a misspelling of, or empty
    patient_id  the patient this sample belongs to, or empty
    note        who decided and why

How a sample's patient is resolved, in order:

0. ``same_as`` rows are applied first: the ID is rewritten everywhere, so the
   two spellings become one sample.
1. ``patient_id`` in the overrides file -> resolved_by = "override"
2. sample table                         -> resolved_by = "sample table"
3. otherwise unresolved. If an ID rule or a near-identical ID suggests a
   patient, it is shown as ``proposed_patient`` but NOT applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from .io import Config, IdRule, Study

ISSUE_COLUMNS = ["level", "code", "entity", "id", "detail"]

# What each code means, shown in the app next to the counts.
CODE_HELP = {
    "PATIENT_CONFLICT": "The sample table and the sample ID name different patients.",
    "TIMELINE_PATIENT_CONFLICT": "The specimen timeline lists the sample under a different patient.",
    "NORMAL_MISMATCH": "The matched normal belongs to a different patient than the tumour sample.",
    "SAMPLE_NOT_IN_TABLE": "The sample appears in the mutation file or timeline but not in the sample table.",
    "NEAR_DUPLICATE_ID": "Two IDs differ only by case, spaces or punctuation.",
    "UNKNOWN_PATIENT": "The sample is assigned to a patient who is not in the patient table.",
    "PATIENT_NOT_IN_TABLE": "The patient has timeline rows but is not in the patient table.",
    "DAY_CONFLICT": "The sample table and the specimen timeline give different collection days.",
    "NO_COLLECTION_DAY": "The sample has no collection day, so it cannot be drawn on the timeline.",
    "NO_MUTATION_CALLS": "The sample has no rows in the mutation file (for example low-pass WGS).",
    "LAB_UNPARSEABLE": "A lab result is not a number or a '<limit' value and is not plotted.",
    "STOP_BEFORE_START": "A treatment's stop day is earlier than its start day.",
    "DAY_UNPARSEABLE": "A treatment day is filled in but is not a number, e.g. a spreadsheet error value.",
    "TREATMENT_ROW_DROPPED": "A treatment row had no drug name or no start day and was dropped.",
    "AGENT_SPELLING": "One drug is written in more than one way; the spellings were merged.",
    "OVERRIDE_APPLIED": "The patient for this sample was set by the overrides file.",
    "ALIAS_APPLIED": "The overrides file says this ID is another spelling of an existing sample.",
}

LEVEL_ORDER = {"error": 0, "warning": 1, "info": 2}


def normalise_id(value: str) -> str:
    """Key for spotting IDs that differ only in formatting: 'pt-001 c1' -> 'PT001C1'."""
    return re.sub(r"[^0-9A-Za-z]", "", value).upper()


def patient_from_sample_id(sample_id: str, rules: tuple[IdRule, ...]) -> str | None:
    for rule in rules:
        patient = rule.patient_for(sample_id)
        if patient:
            return patient
    return None


@dataclass(frozen=True)
class Overrides:
    """Decisions a person has recorded. See the module docstring for the file."""

    patient_of: dict[str, str] = field(default_factory=dict)
    same_as: dict[str, str] = field(default_factory=dict)


def load_overrides(path: Path | None) -> Overrides:
    if path is None:
        return Overrides()
    if not path.exists():
        raise FileNotFoundError(f"Overrides file not found: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame = frame.apply(lambda col: col.str.strip())
    frame = frame[~frame["sample_id"].str.startswith("#") & frame["sample_id"].ne("")]

    def pairs(column: str) -> dict[str, str]:
        if column not in frame.columns:
            return {}
        filled = frame[frame[column].ne("")]
        return dict(zip(filled["sample_id"], filled[column]))

    return Overrides(patient_of=pairs("patient_id"), same_as=pairs("same_as"))


def apply_aliases(study: Study, same_as: dict[str, str]) -> Study:
    """Rewrite misspelt sample IDs to the spelling the overrides file names."""
    if not same_as:
        return study
    return replace(
        study,
        mutations=study.mutations.assign(sample_id=study.mutations["sample_id"].replace(same_as)),
        specimens=study.specimens.assign(sample_id=study.specimens["sample_id"].replace(same_as)),
    )


def _first(frame: pd.DataFrame, key: str, value: str) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.dropna(subset=[key]).drop_duplicates(key).set_index(key)[value].to_dict()


def reconcile(
    study: Study, config: Config, overrides: Overrides | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sample_report, issues).

    sample_report has one row per sample ID seen anywhere in the study.
    issues has one row per problem, with columns level, code, entity, id, detail.
    """
    decisions = overrides or Overrides()
    study = apply_aliases(study, decisions.same_as)
    overrides = decisions.patient_of
    issues: list[tuple[str, str, str, str, str]] = []

    def add(level: str, code: str, entity: str, ident: object, detail: str) -> None:
        issues.append((level, code, entity, str(ident), detail))

    for alias, canonical in sorted(decisions.same_as.items()):
        add("info", "ALIAS_APPLIED", "sample", alias, f"Read as '{canonical}'.")

    table_patient = _first(study.samples, "sample_id", "patient_id")
    table_day = _first(study.samples, "sample_id", "collection_day")
    timeline_patient = _first(study.specimens, "sample_id", "patient_id")
    timeline_day = _first(study.specimens, "sample_id", "day")
    normal_of = _first(study.mutations, "sample_id", "normal_id")
    with_calls = set(study.mutations["sample_id"].dropna())
    known_patients = set(study.patients["patient_id"].dropna())

    all_samples = sorted(set(table_patient) | with_calls | set(timeline_patient) | set(overrides))
    by_normalised: dict[str, list[str]] = {}
    for sample_id in table_patient:
        by_normalised.setdefault(normalise_id(sample_id), []).append(sample_id)

    rows = []
    for sample_id in all_samples:
        in_table = sample_id in table_patient
        from_id = patient_from_sample_id(sample_id, config.id_rules)
        from_timeline = timeline_patient.get(sample_id)
        normal_id = normal_of.get(sample_id)
        normal_patient = (
            patient_from_sample_id(normal_id, config.id_rules) if isinstance(normal_id, str) else None
        )
        proposed = None

        if sample_id in overrides:
            patient, resolved_by = overrides[sample_id], "override"
            add("info", "OVERRIDE_APPLIED", "sample", sample_id, f"Set to {patient}.")
        elif in_table:
            patient, resolved_by = table_patient[sample_id], "sample table"
        else:
            patient, resolved_by = None, "unresolved"
            twins = [s for s in by_normalised.get(normalise_id(sample_id), []) if s != sample_id]
            if twins:
                proposed = table_patient[twins[0]]
                add(
                    "error", "NEAR_DUPLICATE_ID", "sample", sample_id,
                    f"Not in the sample table, but '{twins[0]}' is and differs only in "
                    f"formatting. If they are the same sample, it belongs to {proposed}.",
                )  # fmt: skip
            else:
                proposed = from_id or from_timeline
                where = [
                    name
                    for name, present in (
                        ("mutation file", sample_id in with_calls),
                        ("specimen timeline", sample_id in timeline_patient),
                    )
                    if present
                ]
                hint = f" Its ID suggests {proposed}." if proposed else ""
                add(
                    "error" if sample_id in with_calls else "warning",
                    "SAMPLE_NOT_IN_TABLE", "sample", sample_id,
                    f"Found in the {' and '.join(where)} only.{hint}",
                )  # fmt: skip

        if patient is not None:
            if from_id and from_id != patient and resolved_by == "sample table":
                add(
                    "error", "PATIENT_CONFLICT", "sample", sample_id,
                    f"Sample table says {patient}; the sample ID reads as {from_id}.",
                )  # fmt: skip
            if isinstance(from_timeline, str) and from_timeline != patient:
                add(
                    "error", "TIMELINE_PATIENT_CONFLICT", "sample", sample_id,
                    f"Assigned to {patient}; the specimen timeline says {from_timeline}.",
                )  # fmt: skip
            if normal_patient and normal_patient != patient:
                add(
                    "error", "NORMAL_MISMATCH", "sample", sample_id,
                    f"Tumour is {patient}; matched normal '{normal_id}' reads as {normal_patient}.",
                )  # fmt: skip
            if known_patients and patient not in known_patients:
                add(
                    "error", "UNKNOWN_PATIENT", "sample", sample_id,
                    f"Assigned to {patient}, who is not in the patient table.",
                )  # fmt: skip

        day_table, day_timeline = table_day.get(sample_id), timeline_day.get(sample_id)
        has_table_day, has_timeline_day = pd.notna(day_table), pd.notna(day_timeline)
        if has_table_day and has_timeline_day and day_table != day_timeline:
            add(
                "warning", "DAY_CONFLICT", "sample", sample_id,
                f"Sample table: day {day_table:g}. Specimen timeline: day {day_timeline:g}. "
                "The sample table value is used.",
            )  # fmt: skip
        day = day_table if has_table_day else day_timeline if has_timeline_day else None
        if day is None and patient is not None:
            add(
                "warning" if sample_id in with_calls else "info",
                "NO_COLLECTION_DAY", "sample", sample_id,
                "Variants are listed in the table but not drawn on the timeline.",
            )  # fmt: skip
        if in_table and sample_id not in with_calls:
            add("info", "NO_MUTATION_CALLS", "sample", sample_id, "No rows in the mutation file.")

        rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient,
                "resolved_by": resolved_by,
                "proposed_patient": proposed,
                "collection_day": day,
                "in_sample_table": in_table,
                "in_mutation_file": sample_id in with_calls,
                "in_specimen_timeline": sample_id in timeline_patient,
                "patient_from_id": from_id,
                "patient_in_timeline": from_timeline,
                "normal_id": normal_id,
                "normal_patient": normal_patient,
            }
        )

    # Patients who appear in a timeline file but not in the patient table.
    if known_patients:
        sources = (
            ("treatment timeline", study.treatments, "patient_id"),
            ("lab timeline", study.labs, "patient_id"),
        )
        seen: dict[str, list[str]] = {}
        for name, frame, column in sources:
            for patient in set(frame[column].dropna()) - known_patients:
                seen.setdefault(patient, []).append(name)
        for patient, where in sorted(seen.items()):
            add(
                "warning", "PATIENT_NOT_IN_TABLE", "patient", patient,
                f"Has rows in the {' and '.join(where)}. They are not shown.",
            )  # fmt: skip

    for row in study.labs[study.labs["flag"].eq("unparseable")].itertuples():
        add(
            "warning", "LAB_UNPARSEABLE", "lab result", row.patient_id,
            f"{row.test} on day {row.day:g}: '{row.result_raw}'.",
        )  # fmt: skip

    backwards = study.treatments[study.treatments["stop_day"] < study.treatments["start_day"]]
    for row in backwards.itertuples():
        add(
            "warning", "STOP_BEFORE_START", "treatment", row.patient_id,
            f"{row.agent}: start day {row.start_day:g}, stop day {row.stop_day:g}. "
            "Drawn as a single point at the start day.",
        )  # fmt: skip

    spellings = (
        study.treatments.dropna(subset=["agent_raw"]).groupby("agent")["agent_raw"].unique()
        if not study.treatments.empty
        else pd.Series(dtype=object)
    )
    for agent, variants in spellings.items():
        if len(variants) > 1:
            listed = ", ".join(repr(v) for v in sorted(variants))
            add("info", "AGENT_SPELLING", "drug", agent, f"Merged spellings: {listed}.")

    for code, ident, detail in study.load_notes:
        add("warning", code, "treatment", ident, detail)

    report = pd.DataFrame(rows)
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    issue_frame = issue_frame.sort_values(
        by=["level", "code", "id"], key=lambda col: col.map(LEVEL_ORDER) if col.name == "level" else col
    ).reset_index(drop=True)
    return report, issue_frame


def resolved_samples(study: Study, report: pd.DataFrame) -> pd.DataFrame:
    """The sample table the views should use: one row per resolved sample.

    Patient and collection day come from the reconciliation report, so an
    override in the overrides file reaches every view. Unresolved samples are
    left out; they are listed on the data checks page instead.
    """
    kept = report[report["patient_id"].notna()][["sample_id", "patient_id", "collection_day"]]
    extra = study.samples[["sample_id", "site", "sample_type", "tumor_fraction"]]
    out = kept.merge(extra, on="sample_id", how="left")
    out["collection_day"] = pd.to_numeric(out["collection_day"])
    is_blood = out["sample_id"].str.contains("BLOOD|PLASMA|CFDNA", case=False, regex=True)
    out["sample_type"] = out["sample_type"].fillna(is_blood.map({True: "Blood", False: "Tissue"}))
    return out


def issue_summary(issues: pd.DataFrame) -> pd.DataFrame:
    """Counts per code, most serious first, with a plain-English explanation."""
    if issues.empty:
        return pd.DataFrame(columns=["level", "code", "count", "meaning"])
    summary = issues.groupby(["level", "code"]).size().reset_index(name="count")
    summary["meaning"] = summary["code"].map(CODE_HELP).fillna("")
    order = summary["level"].map(LEVEL_ORDER)
    return summary.assign(_order=order).sort_values(["_order", "code"]).drop(columns="_order")
