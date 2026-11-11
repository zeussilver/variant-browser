"""Read a study directory into tidy tables.

A study directory holds tab-separated files in the cBioPortal layout:

    data_clinical_patient.txt   one row per patient
    data_clinical_sample.txt    one row per sample, with its patient
    data_mutations.txt          one row per variant call (MAF)
    data_timeline_*.txt         dated events: treatments, lab tests, specimens

Only this module knows the source column names. Everything downstream works on
the renamed columns listed in each ``_load_*`` function, so pointing the app at
a different export means editing the mappings here and nothing else.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# Columns read from the MAF. The public file has more than 250 columns and is
# over 80 MB; reading only these keeps start-up to a second or two.
MAF_COLUMNS = {
    "Tumor_Sample_Barcode": "sample_id",
    "Matched_Norm_Sample_Barcode": "normal_id",
    "Hugo_Symbol": "gene",
    "Variant_Classification": "variant_class",
    "HGVSp_Short": "protein_change",
    "Chromosome": "chrom",
    "Start_Position": "pos",
    "Reference_Allele": "ref",
    "Tumor_Seq_Allele2": "alt",
    "t_ref_count": "t_ref",
    "t_alt_count": "t_alt",
}


@dataclass(frozen=True)
class IdRule:
    """One way of reading a patient ID out of a sample ID."""

    pattern: re.Pattern[str]
    template: str

    def patient_for(self, sample_id: str) -> str | None:
        match = self.pattern.search(sample_id)
        return self.template.format(**match.groupdict()) if match else None


@dataclass(frozen=True)
class Config:
    time_axis_label: str
    id_rules: tuple[IdRule, ...]
    gene_panel: tuple[str, ...]
    treatment_classes: dict[str, str]


@dataclass(frozen=True)
class Study:
    name: str
    source_dir: Path
    is_synthetic: bool
    patients: pd.DataFrame
    samples: pd.DataFrame
    mutations: pd.DataFrame
    treatments: pd.DataFrame
    labs: pd.DataFrame
    specimens: pd.DataFrame
    status_events: pd.DataFrame
    # Things the loader had to drop or could not parse, as (code, id, detail).
    load_notes: list[tuple[str, str, str]] = field(default_factory=list)


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    with open(config_dir / "study.toml", "rb") as handle:
        raw = tomllib.load(handle)
    rules = tuple(
        IdRule(re.compile(rule["pattern"], re.IGNORECASE), rule["template"])
        for rule in raw.get("sample_to_patient", [])
    )
    panel = tuple(
        line.strip().upper()
        for line in (config_dir / "gene_panel.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    classes = pd.read_csv(config_dir / "treatment_classes.tsv", sep="\t", dtype=str)
    return Config(
        time_axis_label=raw.get("time_axis_label", "Day"),
        id_rules=rules,
        gene_panel=panel,
        treatment_classes={
            normalise_agent(agent): cls for agent, cls in zip(classes["agent"], classes["class"])
        },
    )


def normalise_agent(name: object) -> str:
    """Collapse spelling variants of a drug name: 'Leuprolide ' -> 'LEUPROLIDE'."""
    if not isinstance(name, str) or not name.strip():
        return ""
    return re.sub(r"\s+", " ", name.strip()).upper()


def _read_tsv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a TSV as text, skipping the '#' description rows cBioPortal puts on top.

    Only leading '#' lines are skipped. A '#' inside a data field is kept, which
    pandas' ``comment='#'`` would silently truncate.
    """
    if not path.exists():
        return pd.DataFrame()
    with open(path, encoding="utf-8") as handle:
        skip = 0
        for line in handle:
            if not line.startswith("#"):
                break
            skip += 1
    frame = pd.read_csv(
        path, sep="\t", dtype=str, skiprows=skip, usecols=usecols, keep_default_na=False
    )
    return frame.apply(lambda col: col.str.strip()).replace("", pd.NA)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """A column by name, or an all-missing column if the file does not have it."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sample_type(sample_id: pd.Series, site: pd.Series) -> pd.Series:
    """'Blood' or 'Tissue', from the recorded site, else from the sample ID."""
    is_blood = site.str.upper().eq("BLOOD").fillna(False) | sample_id.str.contains(
        r"(?:^|[_-])(?:BLOOD|PLASMA|CFDNA)(?:[_-]|$)", case=False, regex=True
    ).fillna(False)
    return is_blood.map({True: "Blood", False: "Tissue"})


def _load_patients(directory: Path) -> pd.DataFrame:
    raw = _read_tsv(directory / "data_clinical_patient.txt")
    if raw.empty:
        return pd.DataFrame(columns=["patient_id"])
    return raw.rename(columns={"PATIENT_ID": "patient_id"})


def _load_samples(directory: Path) -> pd.DataFrame:
    raw = _read_tsv(directory / "data_clinical_sample.txt")
    columns = ["sample_id", "patient_id", "collection_day", "site", "sample_type", "tumor_fraction"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    site = _column(raw, "BX_LOCATION")
    out = pd.DataFrame(
        {
            "sample_id": raw["SAMPLE_ID"],
            "patient_id": raw["PATIENT_ID"],
            "collection_day": _number(_column(raw, "BX_TIME_DAYS")),
            "site": site,
            "sample_type": _sample_type(raw["SAMPLE_ID"], site),
            "tumor_fraction": _number(_column(raw, "TUMOR_FRACTION")),
        }
    )
    return out[columns]


def _load_mutations(directory: Path) -> pd.DataFrame:
    path = directory / "data_mutations.txt"
    columns = [*MAF_COLUMNS.values(), "vaf"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    present = pd.read_csv(path, sep="\t", comment="#", nrows=0).columns
    raw = _read_tsv(path, usecols=[c for c in MAF_COLUMNS if c in present])
    out = raw.rename(columns=MAF_COLUMNS)
    for missing in set(MAF_COLUMNS.values()) - set(out.columns):
        out[missing] = pd.NA
    out["gene"] = out["gene"].str.upper()
    out["pos"] = _number(out["pos"]).astype("Int64")
    out["t_ref"] = _number(out["t_ref"])
    out["t_alt"] = _number(out["t_alt"])
    depth = out["t_ref"] + out["t_alt"]
    # No reads at the site means the allele fraction is unknown, not zero.
    out["vaf"] = (out["t_alt"] / depth).where(depth > 0)
    return out[columns]


def _load_treatments(directory: Path, classes: dict[str, str]) -> pd.DataFrame:
    raw = _read_tsv(directory / "data_timeline_treatment.txt")
    columns = [
        "patient_id", "start_day", "stop_day", "agent_raw", "agent",
        "treatment_type", "setting", "treatment_class",
    ]  # fmt: skip
    if raw.empty:
        return pd.DataFrame(columns=columns)
    treatment_type = _column(raw, "TREATMENT_TYPE")
    agent_raw = _column(raw, "AGENT")
    agent = agent_raw.map(normalise_agent)
    # Radiation rows usually carry no drug name; label them by their type.
    is_radiation = treatment_type.str.upper().str.contains("RADIATION").fillna(False)
    agent = agent.mask(agent.eq("") & is_radiation, "RADIATION THERAPY")
    out = pd.DataFrame(
        {
            "patient_id": raw["PATIENT_ID"],
            "start_day": _number(_column(raw, "START_DATE")),
            "stop_day": _number(_column(raw, "STOP_DATE")),
            "agent_raw": agent_raw,
            "agent": agent,
            "treatment_type": treatment_type,
            "setting": _column(raw, "COMMENTS"),
        }
    )
    out["treatment_class"] = out["agent"].map(classes).fillna("Supportive / other")
    # A day that is filled in but is not a number, e.g. an Excel '#VALUE!'.
    for name, source in (("start_day", "START_DATE"), ("stop_day", "STOP_DATE")):
        out[f"{name}_raw_bad"] = _column(raw, source).where(
            _column(raw, source).notna() & out[name].isna()
        )
    return out[[*columns, "start_day_raw_bad", "stop_day_raw_bad"]]


_LAB_LIMIT = re.compile(r"^([<>])\s*=?\s*([0-9.]+)$")


def parse_lab_result(text: object) -> tuple[float | None, str]:
    """Parse a lab result. Returns (value, flag).

    flag is '' for a plain number, '<' or '>' for a result reported only as
    beyond a limit (the value is that limit), and 'unparseable' otherwise.
    Spaces inside the number are removed first: '<0. 1' is read as '<0.1'.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "unparseable"
    compact = re.sub(r"\s+", "", text)
    try:
        return float(compact), ""
    except ValueError:
        pass
    match = _LAB_LIMIT.match(compact)
    if match:
        try:
            return float(match.group(2)), match.group(1)
        except ValueError:
            pass
    return None, "unparseable"


def _load_labs(directory: Path) -> pd.DataFrame:
    columns = ["patient_id", "day", "test", "result_raw", "value", "flag"]
    frames = []
    for path in sorted(directory.glob("data_timeline_*.txt")):
        raw = _read_tsv(path)
        if raw.empty or "RESULT" not in raw.columns:
            continue
        parsed = raw["RESULT"].map(parse_lab_result)
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": raw["PATIENT_ID"],
                    "day": _number(_column(raw, "START_DATE")),
                    "test": _column(raw, "TEST").fillna(path.stem.removeprefix("data_timeline_")),
                    "result_raw": raw["RESULT"],
                    "value": parsed.map(lambda p: p[0]),
                    "flag": parsed.map(lambda p: p[1]),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    out["test"] = out["test"].str.upper()
    return out[columns]


def _load_specimens(directory: Path) -> pd.DataFrame:
    columns = ["patient_id", "sample_id", "day", "specimen_type", "site"]
    for name in ("data_timeline_sample.txt", "data_timeline_specimen.txt"):
        raw = _read_tsv(directory / name)
        if not raw.empty:
            return pd.DataFrame(
                {
                    "patient_id": raw["PATIENT_ID"],
                    "sample_id": _column(raw, "SAMPLE_ID"),
                    "day": _number(_column(raw, "START_DATE")),
                    "specimen_type": _column(raw, "SPECIMEN_TYPE"),
                    "site": _column(raw, "SPECIMEN_SITE"),
                }
            )[columns]
    return pd.DataFrame(columns=columns)


def _load_status_events(directory: Path) -> pd.DataFrame:
    """Dated one-off events, e.g. diagnosis of metastatic disease."""
    columns = ["patient_id", "day", "label"]
    raw = _read_tsv(directory / "data_timeline_mets.txt")
    if raw.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "patient_id": raw["PATIENT_ID"],
            "day": _number(_column(raw, "START_DATE")),
            "label": "Metastatic diagnosis",
        }
    )[columns]


def _study_name(directory: Path) -> str:
    meta = directory / "meta_study.txt"
    if meta.exists():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip()
    return directory.name


def load_study(directory: Path | str, config: Config | None = None) -> Study:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Study directory not found: {directory}. "
            "Run `python scripts/fetch_data.py` to download the public cohort."
        )
    config = config or load_config()
    notes: list[tuple[str, str, str]] = []

    treatments = _load_treatments(directory, config.treatment_classes)
    empty = treatments["agent"].eq("") | treatments["start_day"].isna()
    for row in treatments[empty].itertuples():
        notes.append(
            ("TREATMENT_ROW_DROPPED", str(row.patient_id), "No drug name or no start day.")
        )
    treatments = treatments[~empty].reset_index(drop=True)
    for column, label in (("start_day_raw_bad", "start"), ("stop_day_raw_bad", "stop")):
        for row in treatments[treatments[column].notna()].itertuples():
            notes.append(
                (
                    "DAY_UNPARSEABLE",
                    str(row.patient_id),
                    f"{row.agent}: {label} day is '{getattr(row, column)}'. Treated as not recorded.",
                )
            )
    treatments = treatments.drop(columns=["start_day_raw_bad", "stop_day_raw_bad"])

    return Study(
        name=_study_name(directory),
        source_dir=directory,
        is_synthetic=(directory / "SYNTHETIC.txt").exists(),
        patients=_load_patients(directory),
        samples=_load_samples(directory),
        mutations=_load_mutations(directory),
        treatments=treatments,
        labs=_load_labs(directory),
        specimens=_load_specimens(directory),
        status_events=_load_status_events(directory),
        load_notes=notes,
    )
