"""The local web app: cohort heatmap, patient timeline and data checks.

Run it with ``variant-browser`` (or ``python -m variant_browser.app``). It
listens on 127.0.0.1 only and loads no scripts from the internet, so the data
stays on the machine it runs on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html, no_update

from . import figures, variants
from .io import CONFIG_DIR, Config, Study, load_config, load_study
from .matching import (
    Overrides,
    apply_aliases,
    issue_summary,
    load_overrides,
    reconcile,
    resolved_samples,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = REPO_ROOT / "data" / "raw" / "mpcproject_broad_2021"
MAX_TABLE_ROWS = 300

LEVEL_MARK = {"error": "✖ Error", "warning": "▲ Warning", "info": "● Note"}


def _table(frame: pd.DataFrame, columns: dict[str, str], limit: int = MAX_TABLE_ROWS) -> html.Div:
    """A plain HTML table. ``columns`` maps column name to header text."""
    shown = frame.head(limit)
    header = html.Tr([html.Th(title) for title in columns.values()])
    body = [
        html.Tr([html.Td("" if pd.isna(v) else str(v)) for v in row])
        for row in shown[list(columns)].itertuples(index=False)
    ]
    children = [html.Table([html.Thead(header), html.Tbody(body)], className="data")]
    if len(frame) > limit:
        children.append(
            html.P(f"Showing the first {limit:,} of {len(frame):,} rows. Download the CSV for all of them.",
                   className="note")
        )  # fmt: skip
    return html.Div(children, className="table-scroll")


def _control(name: str, *children, wide: bool = False) -> html.Div:
    return html.Div(
        [html.Label(name, className="name"), *children],
        className="control wide" if wide else "control",
    )


def _variant_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["patient_id", "collection_day", "gene"]).copy()
    out["Allele fraction"] = out["vaf"].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    out["Reads (alt / total)"] = [
        "" if pd.isna(a) or pd.isna(r) else f"{a:.0f} / {a + r:.0f}"
        for a, r in zip(out["t_alt"], out["t_ref"])
    ]
    out["Collection day"] = out["collection_day"].map(lambda d: "not recorded" if pd.isna(d) else f"{d:,.0f}")
    return out


VARIANT_COLUMNS = {
    "patient_id": "Patient", "sample_type": "Sample type", "Collection day": "Collection day",
    "gene": "Gene", "protein_change": "Protein change", "effect": "Effect",
    "Allele fraction": "Allele fraction", "Reads (alt / total)": "Reads (alt / total)",
    "sample_id": "Sample ID",
}  # fmt: skip


def create_app(study: Study, config: Config, overrides: Overrides | None = None) -> Dash:
    report, issues = reconcile(study, config, overrides)
    # The views read the mutation file too, so they need the corrected IDs.
    study = apply_aliases(study, (overrides or Overrides()).same_as)
    samples = resolved_samples(study, report)
    sequenced_ids = set(study.mutations["sample_id"])
    sequenced = samples[samples["sample_id"].isin(sequenced_ids)]
    # Axis labels are fixed once for all sequenced samples, so a sample keeps
    # its label under every filter and a clicked label maps back to a patient.
    sample_labels = figures.short_sample_labels(sequenced)
    patient_of_label = {
        sample_labels[row.sample_id]: row.patient_id for row in sequenced.itertuples()
    }
    summary = issue_summary(issues)
    n_problems = int(issues["level"].isin(["error", "warning"]).sum())

    # Patients with the most to show come first in the picker.
    per_patient = samples.assign(
        dated_sequenced=samples["sample_id"].isin(sequenced_ids) & samples["collection_day"].notna()
    ).groupby("patient_id").agg(n_samples=("sample_id", "size"), n_dated=("dated_sequenced", "sum"))
    per_patient["n_treatments"] = study.treatments.groupby("patient_id").size().reindex(per_patient.index).fillna(0)
    per_patient = per_patient.sort_values(["n_dated", "n_treatments"], ascending=False)
    patient_options = [
        {
            "label": f"{pid}: {int(r.n_dated)} dated sequenced sample{'s' if r.n_dated != 1 else ''}, "
                     f"{int(r.n_treatments)} treatment records",
            "value": pid,
        }
        for pid, r in per_patient.iterrows()
    ]  # fmt: skip

    app = Dash(__name__, title="Variant browser")
    app.layout = html.Div(
        className="page",
        children=[
            html.H1("Variant browser"),
            html.P(
                f"{study.name}. {sequenced['patient_id'].nunique()} patients with variant calls, "
                f"{len(sequenced)} sequenced samples, {n_problems} data problems to review.",
                className="sub",
            ),
            html.Div(
                "This is synthetic demonstration data. No patient, sample or result in it is real.",
                className="banner",
            ) if study.is_synthetic else None,
            dcc.Tabs(
                id="tabs", value="cohort",
                children=[
                    dcc.Tab(label="Cohort heatmap", value="cohort", children=_cohort_tab(config)),
                    dcc.Tab(label="Patient timeline", value="patient",
                            children=_patient_tab(patient_options)),
                    dcc.Tab(label=f"Data checks ({n_problems})", value="checks",
                            children=_checks_tab(summary)),
                ],
            ),
            dcc.Download(id="download"),
        ],
    )  # fmt: skip

    def filtered_variants(min_vaf_percent, min_alt_reads) -> pd.DataFrame:
        return variants.protein_altering(
            study.mutations, samples,
            min_vaf=(min_vaf_percent or 0) / 100, min_alt_reads=int(min_alt_reads or 0),
        )  # fmt: skip

    def cohort_selection(gene_set, top_n, sample_types, min_vaf, min_reads, order_by):
        calls = filtered_variants(min_vaf, min_reads)
        shown = sequenced[sequenced["sample_type"].isin(sample_types or [])]
        calls = calls[calls["sample_id"].isin(shown["sample_id"])]
        genes = (
            list(config.gene_panel) if gene_set == "panel"
            else variants.most_altered_genes(calls, int(top_n or 25))
        )  # fmt: skip
        sample_ids = shown["sample_id"].tolist()
        grid = variants.gene_sample_grid(calls, genes, sample_ids)
        gene_order, sample_order = variants.order_for_heatmap(grid, genes, sample_ids)
        if order_by == "patient":
            sample_order = shown.sort_values(["patient_id", "sample_type"])["sample_id"].tolist()
        in_view = calls[calls["gene"].isin(genes)]
        return grid, gene_order, sample_order, shown, in_view

    cohort_inputs = [
        Input("gene-set", "value"), Input("top-n", "value"), Input("sample-types", "value"),
        Input("min-vaf", "value"), Input("min-reads", "value"), Input("order-by", "value"),
    ]  # fmt: skip

    @app.callback(
        Output("heatmap", "figure"), Output("cohort-note", "children"),
        Output("cohort-table", "children"), *cohort_inputs,
    )  # fmt: skip
    def update_cohort(*args):
        grid, gene_order, sample_order, shown, in_view = cohort_selection(*args)
        figure = figures.cohort_heatmap(grid, gene_order, sample_order, shown, sample_labels)
        with_hit = grid["sample_id"].nunique()
        note = [
            html.B(f"{with_hit} of {len(sample_order)} samples"),
            f" have a protein-altering variant in at least one of the {len(gene_order)} genes shown. "
            "The percentage after each gene is the share of the samples shown. Copy-number changes "
            "and fusions are not in the mutation file, so they are not shown.",
        ]
        return figure, note, _table(_variant_table(in_view), VARIANT_COLUMNS)

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("download-cohort", "n_clicks"), *[State(i.component_id, "value") for i in cohort_inputs],
        prevent_initial_call=True,
    )  # fmt: skip
    def download_cohort(_clicks, *args):
        in_view = cohort_selection(*args)[4]
        return dcc.send_data_frame(in_view.to_csv, "cohort_variants.csv", index=False)

    @app.callback(
        Output("patient", "value"), Output("tabs", "value"),
        Input("heatmap", "clickData"), prevent_initial_call=True,
    )  # fmt: skip
    def open_patient(click):
        # Dash does not pass a heatmap cell's customdata, so use the column label.
        patient_id = patient_of_label.get(click["points"][0].get("x")) if click else None
        if patient_id is None:
            return no_update, no_update
        return patient_id, "patient"

    def patient_tables(patient_id, min_vaf=0, min_reads=0):
        of_patient = samples[samples["patient_id"] == patient_id]
        calls = filtered_variants(min_vaf, min_reads)
        return of_patient, calls[calls["patient_id"] == patient_id]

    @app.callback(
        Output("timeline", "figure"), Output("patient-note", "children"),
        Output("patient-table", "children"),
        Input("patient", "value"), Input("treatment-classes", "value"), Input("show-other", "value"),
    )  # fmt: skip
    def update_patient(patient_id, classes, show_other):
        if not patient_id:
            return figures._message("Choose a patient."), "", ""
        of_patient, calls = patient_tables(patient_id)
        all_treatments = study.treatments[study.treatments["patient_id"] == patient_id]
        treatments = all_treatments[all_treatments["treatment_class"].isin(classes or [])]
        figure = figures.patient_timeline(
            patient_id, of_patient, calls, treatments,
            study.labs[study.labs["patient_id"] == patient_id],
            study.status_events[study.status_events["patient_id"] == patient_id],
            config.gene_panel, config.time_axis_label, show_other_variants=bool(show_other),
        )  # fmt: skip
        undated = of_patient[of_patient["collection_day"].isna()]
        note = [
            html.B(f"{len(of_patient)} sample{'s' if len(of_patient) != 1 else ''}"),
            f", {int(of_patient['sample_id'].isin(sequenced_ids).sum())} with variant calls. ",
        ]
        hidden = len(all_treatments) - len(treatments)
        if hidden:
            note.append(
                f"{hidden} treatment record{'s are' if hidden != 1 else ' is'} hidden by the class filter. "
            )
        if not undated.empty:
            note.append(
                f"{len(undated)} sample{'s have' if len(undated) != 1 else ' has'} no collection day and "
                "cannot be drawn; the variants are in the table below."
            )
        in_panel = calls["gene"].isin(config.gene_panel)
        table_rows = calls if show_other else calls[in_panel]
        columns = {k: v for k, v in VARIANT_COLUMNS.items() if k != "patient_id"}
        return figure, note, _table(_variant_table(table_rows), columns)

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("download-patient", "n_clicks"), State("patient", "value"), prevent_initial_call=True,
    )  # fmt: skip
    def download_patient(_clicks, patient_id):
        if not patient_id:
            return no_update
        calls = patient_tables(patient_id)[1]
        return dcc.send_data_frame(calls.to_csv, f"{patient_id}_variants.csv", index=False)

    @app.callback(Output("issue-table", "children"), Input("issue-code", "value"))
    def update_issues(code):
        shown = issues if code in (None, "all") else issues[issues["code"] == code]
        shown = shown.assign(level=shown["level"].map(LEVEL_MARK))
        return _table(
            shown, {"level": "Level", "code": "Check", "entity": "About", "id": "ID", "detail": "What was found"}
        )  # fmt: skip

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("download-issues", "n_clicks"), prevent_initial_call=True,
    )  # fmt: skip
    def download_issues(_clicks):
        return dcc.send_data_frame(issues.to_csv, "data_issues.csv", index=False)

    @app.callback(
        Output("download", "data", allow_duplicate=True),
        Input("download-report", "n_clicks"), prevent_initial_call=True,
    )  # fmt: skip
    def download_report(_clicks):
        return dcc.send_data_frame(report.to_csv, "sample_patient_report.csv", index=False)

    return app


def _cohort_tab(config: Config) -> list:
    return [
        html.Div(
            className="controls",
            children=[
                _control(
                    "Genes",
                    dcc.RadioItems(
                        id="gene-set", value="panel", inline=True, className="inline",
                        options=[
                            {"label": f"Gene panel ({len(config.gene_panel)} genes)", "value": "panel"},
                            {"label": "Most often altered", "value": "top"},
                        ],
                    ),
                ),
                _control("How many (most often altered)", dcc.Input(id="top-n", type="number", value=25, min=5, max=60)),
                _control(
                    "Sample type",
                    dcc.Checklist(id="sample-types", options=["Tissue", "Blood"], value=["Tissue", "Blood"],
                                  inline=True, className="inline"),
                ),
                _control("Minimum allele fraction (%)", dcc.Input(id="min-vaf", type="number", value=0, min=0, max=100, step=0.5)),
                _control("Minimum variant reads", dcc.Input(id="min-reads", type="number", value=0, min=0, step=1)),
                _control(
                    "Order samples by",
                    dcc.RadioItems(
                        id="order-by", value="pattern", inline=True, className="inline",
                        options=[{"label": "Mutation pattern", "value": "pattern"},
                                 {"label": "Patient", "value": "patient"}],
                    ),
                ),
            ],
        ),
        html.P(id="cohort-note", className="note"),
        html.Div(dcc.Graph(id="heatmap", config={"displaylogo": False}), className="card"),
        html.H2("Variants in view"),
        html.Button("Download as CSV", id="download-cohort", className="plain"),
        html.Div(id="cohort-table", className="card"),
    ]  # fmt: skip


def _patient_tab(patient_options: list[dict]) -> list:
    shown_by_default = [c for c in figures.TREATMENT_CLASSES if c != "Supportive / other"]
    return [
        html.Div(
            className="controls",
            children=[
                _control(
                    "Patient",
                    dcc.Dropdown(id="patient", options=patient_options, clearable=False,
                                 value=patient_options[0]["value"] if patient_options else None,
                                 style={"width": "560px"}),
                    wide=True,
                ),
                _control(
                    "Treatment classes",
                    dcc.Checklist(id="treatment-classes", options=list(figures.TREATMENT_CLASSES),
                                  value=shown_by_default, inline=True, className="inline"),
                ),
                _control(
                    "Variants",
                    dcc.Checklist(id="show-other", options=[{"label": "Show variants outside the gene panel", "value": "yes"}],
                                  value=[], className="inline"),
                ),
            ],
        ),
        html.P(id="patient-note", className="note"),
        html.Div(dcc.Graph(id="timeline", config={"displaylogo": False}), className="card"),
        html.H2("This patient's variants"),
        html.Button("Download as CSV", id="download-patient", className="plain"),
        html.Div(id="patient-table", className="card"),
    ]  # fmt: skip


def _checks_tab(summary: pd.DataFrame) -> list:
    shown = summary.assign(level=summary["level"].map(LEVEL_MARK))
    options = [{"label": "All checks", "value": "all"}] + [
        {"label": f"{row.code} ({row.count})", "value": row.code} for row in summary.itertuples()
    ]
    return [
        html.P(
            "Every sample and patient ID is compared across the sample table, the mutation file and "
            "the timeline files. Nothing is corrected automatically: a sample whose patient is in "
            "doubt is listed here, and a person records the decision in the overrides file.",
            className="note",
        ),
        html.Div(
            _table(shown, {"level": "Level", "code": "Check", "count": "Count", "meaning": "What it means"}),
            className="card",
        ),
        html.H2("Details"),
        html.Div(
            className="controls",
            children=[
                _control("Show", dcc.Dropdown(id="issue-code", options=options, value="all", clearable=False,
                                              style={"width": "360px"}), wide=True),
                html.Div([
                    html.Button("Download all issues (CSV)", id="download-issues", className="plain"),
                    html.Button("Download sample-to-patient report (CSV)", id="download-report", className="plain"),
                ]),
            ],
        ),
        html.Div(id="issue-table", className="card"),
    ]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY, help="study directory to load")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR, help="configuration directory")
    parser.add_argument("--overrides", type=Path, default=None,
                        help="TSV with columns sample_id, same_as, patient_id, note")  # fmt: skip
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="reload on code changes")
    args = parser.parse_args()

    config = load_config(args.config)
    study = load_study(args.study, config)
    app = create_app(study, config, load_overrides(args.overrides))
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
