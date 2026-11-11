"""Filter variant calls and shape them for the two views."""

from __future__ import annotations

import pandas as pd

# Variant classes that change the protein, grouped into the three effects the
# views colour by. Anything not listed (silent, intron, UTR, ...) is left out.
EFFECT_OF_CLASS = {
    "Missense_Mutation": "Missense",
    "Nonsense_Mutation": "Truncating",
    "Frame_Shift_Ins": "Truncating",
    "Frame_Shift_Del": "Truncating",
    "Splice_Site": "Truncating",
    "Nonstop_Mutation": "Truncating",
    "Translation_Start_Site": "Truncating",
    "In_Frame_Ins": "In-frame indel",
    "In_Frame_Del": "In-frame indel",
}
# Most damaging first. A gene with several hits in one sample takes the colour
# of its most damaging one.
EFFECTS = ("Truncating", "In-frame indel", "Missense")
_SEVERITY = {effect: rank for rank, effect in enumerate(EFFECTS)}

VARIANT_KEY = ["chrom", "pos", "ref", "alt"]


def protein_altering(
    mutations: pd.DataFrame,
    samples: pd.DataFrame,
    min_vaf: float = 0.0,
    min_alt_reads: int = 0,
) -> pd.DataFrame:
    """Protein-altering calls in resolved samples, with patient and effect attached.

    A call with unknown allele fraction (no reads at the site) is kept unless a
    minimum allele fraction above zero is asked for.
    """
    out = mutations.assign(effect=mutations["variant_class"].map(EFFECT_OF_CLASS))
    out = out[out["effect"].notna()]
    if min_vaf > 0:
        out = out[out["vaf"] >= min_vaf]
    if min_alt_reads > 0:
        out = out[out["t_alt"] >= min_alt_reads]
    out = out.merge(
        samples[["sample_id", "patient_id", "collection_day", "sample_type"]],
        on="sample_id",
        how="inner",
    )
    out["label"] = out["gene"] + " " + out["protein_change"].fillna("(no protein change)")
    out["variant_key"] = out[VARIANT_KEY].astype(str).agg(":".join, axis=1)
    return out.reset_index(drop=True)


def most_altered_genes(variants: pd.DataFrame, top_n: int) -> list[str]:
    """Genes altered in the most patients. Ties broken alphabetically."""
    counts = variants.groupby("gene")["patient_id"].nunique()
    ranked = counts.reset_index().sort_values(["patient_id", "gene"], ascending=[False, True])
    return ranked["gene"].head(top_n).tolist()


def gene_sample_grid(variants: pd.DataFrame, genes: list[str], sample_ids: list[str]) -> pd.DataFrame:
    """One row per (gene, sample) that has at least one variant.

    Columns: gene, sample_id, effect (most damaging), n_variants, changes
    (protein changes joined), max_vaf.
    """
    columns = ["gene", "sample_id", "effect", "n_variants", "changes", "max_vaf"]
    subset = variants[variants["gene"].isin(genes) & variants["sample_id"].isin(sample_ids)]
    if subset.empty:
        return pd.DataFrame(columns=columns)
    subset = subset.assign(_severity=subset["effect"].map(_SEVERITY)).sort_values("_severity")
    grid = subset.groupby(["gene", "sample_id"], sort=False).agg(
        effect=("effect", "first"),
        n_variants=("effect", "size"),
        changes=("protein_change", lambda s: ", ".join(s.fillna("?"))),
        max_vaf=("vaf", "max"),
    )
    return grid.reset_index()[columns]


def order_for_heatmap(
    grid: pd.DataFrame, genes: list[str], sample_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Row and column order that makes the pattern readable.

    Genes: most often altered first (then the order given). Samples: sorted so
    that those altered in the first gene come first, then within them those
    altered in the second gene, and so on. Samples with no hits end up last.
    """
    altered = grid.groupby("gene")["sample_id"].nunique()
    gene_order = sorted(genes, key=lambda g: (-altered.get(g, 0), genes.index(g)))
    hits = set(zip(grid["gene"], grid["sample_id"]))
    sample_order = sorted(
        sample_ids, key=lambda s: tuple(0 if (g, s) in hits else 1 for g in gene_order) + (s,)
    )
    return gene_order, sample_order


def serial_variant_keys(patient_variants: pd.DataFrame) -> set[str]:
    """Variants seen in more than one dated sample of a patient."""
    dated = patient_variants[patient_variants["collection_day"].notna()]
    counts = dated.groupby("variant_key")["sample_id"].nunique()
    return set(counts[counts > 1].index)
