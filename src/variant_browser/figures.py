"""Build the two Plotly figures. Pure functions: tables in, figure out.

Nothing here reads files or knows about Dash, so the figures can be built and
checked in a test or a notebook.

Colour rules used throughout:
* A variant's effect always has the same colour in both views.
* Text is never coloured; a coloured mark next to it carries the meaning.
* Two measures on different scales get two panels, never a second y-axis.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .variants import EFFECTS, serial_variant_keys

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EMPTY_CELL = "#f0efec"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Checked with the dataviz palette validator: these three stay distinguishable
# for every pair, including under colour-vision deficiency.
EFFECT_COLOUR = {"Missense": "#2a78d6", "Truncating": "#eb6834", "In-frame indel": "#1baf7a"}
_EFFECT_CODE = {"Missense": 1, "In-frame indel": 2, "Truncating": 3}

SAMPLE_TYPE_COLOUR = {"Tissue": "#c3c2b7", "Blood": "#52514e"}

# Fixed order. A class keeps its colour whichever classes are switched on.
TREATMENT_CLASSES = (
    "Hormonal",
    "AR pathway inhibitor",
    "Chemotherapy",
    "Radiation / radioligand",
    "Targeted / immune",
    "Clinical trial",
    "Supportive / other",
)
TREATMENT_COLOUR = dict(
    zip(
        TREATMENT_CLASSES,
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", INK_MUTED),
    )
)


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=12, color=INK)),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0, font=dict(color=INK_SECONDARY)),
        margin=dict(l=10, r=20, t=30, b=10),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True)
    return fig


def _message(text: str, height: int = 220) -> go.Figure:
    """An empty figure that says why there is nothing to draw."""
    fig = go.Figure()
    fig.add_annotation(
        text=text, showarrow=False, font=dict(size=14, color=INK_SECONDARY, family=FONT)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height)


def _legend_key(name: str, colour: str, symbol: str = "square", group: str | None = None):
    """A legend entry with no data behind it (heatmaps and shapes have no legend)."""
    return go.Scatter(
        x=[None], y=[None], mode="markers", name=name, legendgroup=group or name,
        marker=dict(symbol=symbol, size=11, color=colour), hoverinfo="skip",
    )  # fmt: skip


def short_sample_labels(samples: pd.DataFrame) -> dict[str, str]:
    """Compact axis labels: the part of the patient ID that varies, plus T or B.

    'MPCPROJECT_0013' tissue -> '0013 T'. A second sample of the same type for
    the same patient gets a number: '0013 T2'.
    """
    patients = samples["patient_id"].astype(str).tolist()
    prefix = os.path.commonprefix(patients) if len(set(patients)) > 1 else ""
    labels = {}
    ordered = samples.sort_values(["patient_id", "sample_type", "collection_day", "sample_id"])
    counter: dict[tuple[str, str], int] = {}
    for row in ordered.itertuples():
        key = (row.patient_id, row.sample_type)
        counter[key] = counter.get(key, 0) + 1
        suffix = str(row.sample_type)[0] + (str(counter[key]) if counter[key] > 1 else "")
        labels[row.sample_id] = f"{str(row.patient_id)[len(prefix):]} {suffix}"
    return labels


def cohort_heatmap(
    grid: pd.DataFrame,
    gene_order: list[str],
    sample_order: list[str],
    samples: pd.DataFrame,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Genes down, samples across. A cell is coloured by the most damaging variant.

    A number in a cell means that many variants in that gene in that sample.
    ``labels`` maps sample ID to its axis label. Pass the same mapping for every
    filter setting so a sample keeps its label, and so the app can turn a
    clicked label back into a patient.
    """
    if not gene_order or not sample_order:
        return _message("No samples match the current filters.")

    info = samples.set_index("sample_id")
    labels = labels or short_sample_labels(samples[samples["sample_id"].isin(sample_order)])
    cell = {(row.gene, row.sample_id): row for row in grid.itertuples()}
    x_labels = [labels[s] for s in sample_order]

    z, text, custom = [], [], []
    for gene in gene_order:
        z_row, text_row, custom_row = [], [], []
        for sample_id in sample_order:
            hit = cell.get((gene, sample_id))
            patient = info.at[sample_id, "patient_id"]
            sample_type = info.at[sample_id, "sample_type"]
            if hit is None:
                z_row.append(0)
                text_row.append("")
                detail = "No protein-altering variant"
            else:
                z_row.append(_EFFECT_CODE[hit.effect])
                text_row.append(str(hit.n_variants) if hit.n_variants > 1 else "")
                vaf = f"{hit.max_vaf:.1%}" if pd.notna(hit.max_vaf) else "unknown"
                detail = f"{hit.effect}: {hit.changes}<br>Highest allele fraction: {vaf}"
            custom_row.append([patient, sample_id, sample_type, gene, detail])
        z.append(z_row)
        text.append(text_row)
        custom.append(custom_row)

    n_samples = len(sample_order)
    altered = grid.groupby("gene")["sample_id"].nunique()
    y_labels = [f"{gene}  {altered.get(gene, 0) / n_samples:.0%}" for gene in gene_order]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.012,
        row_heights=[18, 26 * len(gene_order)],
    )  # fmt: skip
    type_codes = [[0 if info.at[s, "sample_type"] == "Tissue" else 1 for s in sample_order]]
    fig.add_trace(
        go.Heatmap(
            z=type_codes, x=x_labels, y=["Sample type"], zmin=0, zmax=1, showscale=False,
            colorscale=[[0, SAMPLE_TYPE_COLOUR["Tissue"]], [1, SAMPLE_TYPE_COLOUR["Blood"]]],
            xgap=2, ygap=2,
            customdata=[[[info.at[s, "patient_id"], s, info.at[s, "sample_type"]] for s in sample_order]],
            hovertemplate="%{customdata[0]}<br>%{customdata[2]} sample<extra></extra>",
        ),
        row=1, col=1,
    )  # fmt: skip
    steps = [EMPTY_CELL, EFFECT_COLOUR["Missense"], EFFECT_COLOUR["In-frame indel"], EFFECT_COLOUR["Truncating"]]
    colourscale = []
    for index, colour in enumerate(steps):
        colourscale += [[index / 4, colour], [(index + 1) / 4, colour]]
    fig.add_trace(
        go.Heatmap(
            z=z, x=x_labels, y=y_labels, zmin=-0.5, zmax=3.5, showscale=False,
            colorscale=colourscale, xgap=2, ygap=2,
            text=text, texttemplate="%{text}", textfont=dict(color="white", size=11),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[3]}</b> in %{customdata[0]}<br>%{customdata[2]} sample"
                "<br>%{customdata[4]}<br><i>Click to open this patient</i><extra></extra>"
            ),
        ),
        row=2, col=1,
    )  # fmt: skip
    for effect in EFFECTS:
        fig.add_trace(_legend_key(effect, EFFECT_COLOUR[effect]))
    for sample_type, colour in SAMPLE_TYPE_COLOUR.items():
        fig.add_trace(_legend_key(f"{sample_type} sample", colour))

    _base_layout(fig, height=26 * len(gene_order) + 150)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK), row=2, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, showline=False)
    # Past about 100 columns the tick labels are unreadable; hover names the sample.
    fig.update_xaxes(showticklabels=n_samples <= 100, tickangle=-90, tickfont=dict(size=9), row=2, col=1)
    fig.update_layout(legend=dict(y=1.0, yanchor="bottom"), margin=dict(t=44))
    return fig


def _treatment_rows(treatments: pd.DataFrame) -> list[str]:
    """Drug rows, grouped by class in the fixed class order, earliest start first."""
    first = treatments.groupby("agent").agg(cls=("treatment_class", "first"), start=("start_day", "min"))
    first["_rank"] = first["cls"].map({c: i for i, c in enumerate(TREATMENT_CLASSES)})
    return first.sort_values(["_rank", "start"]).index.tolist()


def _day(value: float) -> str:
    return f"day {value:,.0f}"


def patient_timeline(
    patient_id: str,
    samples: pd.DataFrame,
    variants: pd.DataFrame,
    treatments: pd.DataFrame,
    labs: pd.DataFrame,
    status_events: pd.DataFrame,
    gene_panel: tuple[str, ...],
    time_axis_label: str = "Day",
    show_other_variants: bool = False,
) -> go.Figure:
    """One patient: treatments, variant allele fractions and markers on one time axis.

    Every argument is already filtered to this patient. ``variants`` is the
    output of ``variants.protein_altering``. Panels with no data are left out.
    """
    dated_samples = samples[samples["collection_day"].notna()].sort_values("collection_day")
    dated_variants = variants[variants["collection_day"].notna()]
    fractions = dated_samples[dated_samples["tumor_fraction"].notna()]
    plotted_labs = labs[labs["value"].notna() & labs["day"].notna()].sort_values("day")
    lab_tests = sorted(plotted_labs["test"].unique())

    panels: list[tuple[str, str, float]] = []  # (key, title, relative height)
    if not treatments.empty:
        n_rows = treatments["agent"].nunique()
        panels.append(("treatments", "Treatments", max(1.2, 0.34 * n_rows + 0.5)))
    if not dated_variants.empty:
        panels.append(("variants", "Variant allele fraction in each sequenced sample", 2.4))
    if not fractions.empty:
        panels.append(("fraction", "ctDNA tumour fraction", 1.3))
    for test in lab_tests:
        panels.append((f"lab:{test}", f"{test} (log scale)", 1.8))
    if not panels:
        return _message(f"No dated treatments, samples or lab results for {patient_id}.")

    fig = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[height for _, _, height in panels],
        subplot_titles=[title for _, title, _ in panels],
    )  # fmt: skip
    row_of = {key: index + 1 for index, (key, _, _) in enumerate(panels)}

    day_columns = [
        treatments["start_day"], treatments["stop_day"], dated_samples["collection_day"],
        plotted_labs["day"], status_events["day"],
    ]  # fmt: skip
    all_days = pd.concat([c.astype(float) for c in day_columns if not c.empty]).dropna()
    first_day, last_day = float(all_days.min()), float(all_days.max())
    span = max(last_day - first_day, 1.0)

    if "treatments" in row_of:
        _add_treatments(fig, treatments, row_of["treatments"], last_day, span)
    if "variants" in row_of:
        _add_variants(fig, dated_variants, gene_panel, row_of["variants"], show_other_variants)
    if "fraction" in row_of:
        fig.add_trace(
            go.Scatter(
                x=fractions["collection_day"], y=fractions["tumor_fraction"],
                mode="lines+markers" if len(fractions) > 1 else "markers",
                name="ctDNA tumour fraction", showlegend=False,
                line=dict(color=INK_SECONDARY, width=2),
                marker=dict(size=9, color=INK_SECONDARY, line=dict(color=SURFACE, width=2)),
                hovertemplate="Tumour fraction %{y:.1%}<br>day %{x:,.0f}<extra></extra>",
            ),
            row=row_of["fraction"], col=1,
        )  # fmt: skip
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=row_of["fraction"], col=1)
    for test in lab_tests:
        _add_lab(fig, plotted_labs[plotted_labs["test"] == test], test, row_of[f"lab:{test}"])

    # Sample collection days run through every panel, so a variant can be read
    # against the treatment above it and the marker below it.
    for index, row in enumerate(dated_samples.itertuples()):
        fig.add_vline(x=row.collection_day, line=dict(color=INK_MUTED, width=1, dash="dot"))
        sequenced = row.sample_id in set(variants["sample_id"])
        fig.add_annotation(
            x=row.collection_day, yref="paper", y=1.0, yanchor="bottom", showarrow=False,
            yshift=22 + 13 * (index % 2), font=dict(size=10, color=INK_SECONDARY),
            text=f"{row.sample_type} sample" + ("" if sequenced else " (no variant calls)"),
        )  # fmt: skip
    for row in status_events.dropna(subset=["day"]).itertuples():
        fig.add_vline(x=row.day, line=dict(color=INK, width=1, dash="dash"))
        fig.add_annotation(
            x=row.day, yref="paper", y=0.0, yanchor="top", showarrow=False, yshift=-22,
            font=dict(size=10, color=INK), text=row.label,
        )  # fmt: skip

    height = int(150 + 95 * sum(h for _, _, h in panels))
    _base_layout(fig, height)
    # The right margin holds the variant labels.
    fig.update_layout(barmode="overlay", margin=dict(t=78, b=30, r=190), legend=dict(y=-0.1 - 40 / height))
    for annotation in fig.layout.annotations[: len(panels)]:  # the panel titles
        annotation.update(x=0, xanchor="left", font=dict(size=13, color=INK))
    fig.update_xaxes(range=[first_day - 0.03 * span, last_day + 0.05 * span])
    fig.update_xaxes(title_text=time_axis_label, row=len(panels), col=1)
    return fig


def _add_treatments(fig: go.Figure, treatments: pd.DataFrame, row: int, last_day: float, span: float) -> None:
    order = _treatment_rows(treatments)
    names = {agent: agent.capitalize() for agent in order}
    frame = treatments.copy()
    frame["open_ended"] = frame["stop_day"].isna()
    frame["end"] = frame["stop_day"].fillna(last_day)
    # Too short to see as a bar, or stop recorded before start: draw one point.
    frame["is_point"] = (frame["end"] - frame["start_day"]) < 0.006 * span
    frame["stop_text"] = [
        "stop day not recorded" if r.open_ended
        else f"stop day {r.stop_day:,.0f} is before the start" if r.stop_day < r.start_day
        else f"to {_day(r.stop_day)}"
        for r in frame.itertuples()
    ]  # fmt: skip
    hover = (
        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>from day %{base:,.0f}, %{customdata[2]}"
        "<br>Setting: %{customdata[3]}<extra></extra>"
    )
    for cls in TREATMENT_CLASSES:
        of_class = frame[frame["treatment_class"] == cls]
        if of_class.empty:
            continue
        colour = TREATMENT_COLOUR[cls]
        shown_in_legend = False
        for open_ended in (False, True):
            bars = of_class[~of_class["is_point"] & (of_class["open_ended"] == open_ended)]
            if bars.empty:
                continue
            fig.add_trace(
                go.Bar(
                    y=bars["agent"].map(names), base=bars["start_day"], x=bars["end"] - bars["start_day"],
                    orientation="h", width=0.56, name=cls, legendgroup=cls,
                    showlegend=not shown_in_legend,
                    marker=dict(
                        color=colour, cornerradius=4, line=dict(color=SURFACE, width=1),
                        pattern=dict(shape="/" if open_ended else "", fgcolor=SURFACE, size=6),
                    ),
                    customdata=bars[["agent", "treatment_class", "stop_text", "setting"]].fillna("not recorded"),
                    hovertemplate=hover,
                ),
                row=row, col=1,
            )  # fmt: skip
            shown_in_legend = True
        points = of_class[of_class["is_point"]]
        if not points.empty:
            fig.add_trace(
                go.Scatter(
                    y=points["agent"].map(names), x=points["start_day"], mode="markers",
                    name=cls, legendgroup=cls, showlegend=not shown_in_legend,
                    marker=dict(symbol="diamond", size=10, color=colour, line=dict(color=SURFACE, width=2)),
                    customdata=points[["agent", "treatment_class", "stop_text", "setting"]].fillna("not recorded"),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>day %{x:,.0f}, %{customdata[2]}"
                        "<br>Setting: %{customdata[3]}<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )  # fmt: skip
    if frame["open_ended"].any():
        fig.add_trace(
            go.Bar(
                x=[None], y=[None], name="Stop day not recorded (drawn to the last record)",
                marker=dict(color=INK_MUTED, pattern=dict(shape="/", fgcolor=SURFACE, size=6)),
                hoverinfo="skip",
            ),
            row=row, col=1,
        )  # fmt: skip
    fig.update_yaxes(
        categoryorder="array", categoryarray=[names[a] for a in reversed(order)],
        showgrid=False, tickfont=dict(color=INK), row=row, col=1,
    )  # fmt: skip


def _add_variants(
    fig: go.Figure,
    variants: pd.DataFrame,
    gene_panel: tuple[str, ...],
    row: int,
    show_other: bool,
) -> None:
    in_panel = variants["gene"].isin(gene_panel)
    hover = (
        "<b>%{customdata[0]}</b><br>Allele fraction %{y:.1%} (%{customdata[1]:.0f} of "
        "%{customdata[2]:.0f} reads)<br>%{customdata[3]} sample, day %{x:,.0f}<extra></extra>"
    )
    plotted = variants.assign(depth=variants["t_alt"] + variants["t_ref"])
    plotted = plotted[plotted["vaf"].notna()].sort_values("collection_day")
    custom_columns = ["label", "t_alt", "depth", "sample_type"]

    others = plotted[~in_panel.reindex(plotted.index)]
    if show_other and not others.empty:
        fig.add_trace(
            go.Scatter(
                x=others["collection_day"], y=others["vaf"], mode="markers",
                name="Variants outside the gene panel",
                marker=dict(size=6, color=INK_MUTED, opacity=0.45),
                customdata=others[custom_columns], hovertemplate=hover,
            ),
            row=row, col=1,
        )  # fmt: skip

    panel = plotted[in_panel.reindex(plotted.index)]
    tracked = serial_variant_keys(panel)
    for effect in EFFECTS:
        of_effect = panel[panel["effect"] == effect]
        if of_effect.empty:
            continue
        colour = EFFECT_COLOUR[effect]
        # One line per variant seen in more than one sample; gaps (None) separate them.
        xs, ys, custom = [], [], []
        # Tissue and plasma allele fractions measure different things, so a line
        # only joins samples of the same type.
        for _, track in of_effect[of_effect["variant_key"].isin(tracked)].groupby(
            ["variant_key", "sample_type"], sort=False
        ):
            if len(track) < 2:
                continue
            xs += [*track["collection_day"], None]
            ys += [*track["vaf"], None]
            custom += [*track[custom_columns].values.tolist(), [None] * 4]
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", line=dict(color=colour, width=2),
                    legendgroup=effect, showlegend=False, hoverinfo="skip", connectgaps=False,
                ),
                row=row, col=1,
            )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=of_effect["collection_day"], y=of_effect["vaf"], mode="markers",
                name=effect, legendgroup=effect,
                marker=dict(
                    size=9, color=colour, line=dict(color=SURFACE, width=2),
                    symbol=["square" if t == "Tissue" else "circle" for t in of_effect["sample_type"]],
                ),
                customdata=of_effect[custom_columns], hovertemplate=hover,
            ),
            row=row, col=1,
        )  # fmt: skip
    _label_variants(fig, panel, row)
    if (panel["sample_type"] == "Tissue").any() and (panel["sample_type"] == "Blood").any():
        fig.add_trace(_legend_key("Tissue sample", INK_SECONDARY, "square"), row=row, col=1)
        fig.add_trace(_legend_key("Blood sample", INK_SECONDARY, "circle"), row=row, col=1)
    if panel.empty and not (show_other and not others.empty):
        fig.add_annotation(
            text="No variants in the gene panel. Tick 'Show variants outside the gene panel' to see the rest.",
            xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False,
            font=dict(color=INK_SECONDARY), row=row, col=1,
        )  # fmt: skip
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=row, col=1)


def spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Positions for labels so that none is closer than ``min_gap`` to the next.

    Values must be sorted from high to low. Each label stays as near its value
    as the gap allows, and the group stays centred on where it started.
    """
    if not values:
        return []
    placed = [values[0]]
    for value in values[1:]:
        placed.append(min(value, placed[-1] - min_gap))
    shift = (sum(values) - sum(placed)) / len(values)
    return [position + shift for position in placed]


def _label_variants(fig: go.Figure, panel: pd.DataFrame, row: int, per_sample: int = 8) -> None:
    """Name each panel variant once, at its last sample, with a short leader line.

    Variants from the same clone have nearly the same allele fraction, so their
    labels would print on top of each other. They are spread apart instead.
    """
    if panel.empty:
        return
    y_axis = fig.get_subplot(row, 1).yaxis.plotly_name.replace("axis", "")
    x_axis = fig.get_subplot(row, 1).xaxis.plotly_name.replace("axis", "")
    top = panel["vaf"].max() * 1.08
    last_day = panel.groupby("variant_key")["collection_day"].transform("max")
    at_last = panel[panel["collection_day"] == last_day]
    for _, group in at_last.groupby("sample_id"):
        group = group.sort_values("vaf", ascending=False).head(per_sample)
        positions = spread_labels(group["vaf"].tolist(), min_gap=0.085 * top)
        for variant, position in zip(group.itertuples(), positions):
            fig.add_annotation(
                x=variant.collection_day, y=variant.vaf, xref=x_axis, yref=y_axis,
                ax=26, axref="pixel", ay=position, ayref=y_axis,
                text=variant.label, xanchor="left", showarrow=True, arrowhead=0,
                arrowwidth=1, arrowcolor=AXIS, standoff=5,
                font=dict(size=10, color=INK_SECONDARY),
            )  # fmt: skip
    fig.update_yaxes(range=[-0.04 * top, top], row=row, col=1)


def _add_lab(fig: go.Figure, results: pd.DataFrame, test: str, row: int) -> None:
    """One lab test on a log axis. Results below a detection limit are open circles."""
    positive = results.loc[results["value"] > 0, "value"]
    floor = min(0.01, positive.min()) if not positive.empty else 0.01
    is_zero = results["value"] <= 0
    y = results["value"].where(~is_zero, floor)
    below_limit = results["flag"].isin(["<", ">"]) | is_zero
    note = [
        "reported as 0, drawn at the axis floor" if zero
        else f"reported as '{raw}'" if flagged
        else ""
        for zero, flagged, raw in zip(is_zero, below_limit, results["result_raw"])
    ]  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=results["day"], y=y, mode="lines+markers", name=test, showlegend=False,
            line=dict(color=INK_SECONDARY, width=2),
            marker=dict(
                size=8, color=[SURFACE if b else INK_SECONDARY for b in below_limit],
                line=dict(color=[INK_SECONDARY if b else SURFACE for b in below_limit], width=2),
            ),
            customdata=list(zip(results["result_raw"], note)),
            hovertemplate=f"{test} %{{customdata[0]}}<br>day %{{x:,.0f}}<br>%{{customdata[1]}}<extra></extra>",
        ),
        row=row, col=1,
    )  # fmt: skip
    fig.update_yaxes(type="log", row=row, col=1)
