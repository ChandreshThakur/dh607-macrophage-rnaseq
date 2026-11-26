"""Phase 5 adversarial validation of the macrophage interaction findings."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from preprocessing.normalization import log2_transform, median_of_ratios
from statistical_analysis.interaction import benjamini_hochberg, interaction_contrasts


FDR = 0.05
IFNL_SYMBOLS = ("IFNL1", "IFNL2", "IFNL3")
NEGATIVE_SYMBOLS = (
    "CXCL10", "IDO1", "GBP1", "GBP2", "GBP3", "GBP4", "GBP5",
    "TNF", "IL1B", "SOCS3", "TNFAIP3",
)
PROGRAMS = {
    "antimicrobial_GBP": ("GBP1", "GBP2", "GBP3", "GBP4", "GBP5"),
    "classical_interferon": (
        "STAT1", "IRF1", "IDO1", "CXCL9", "CXCL10", "CXCL11",
        "IFIT1", "IFIT2", "IFIT3", "ISG15", "OAS1",
    ),
    "inflammatory": ("TNF", "IL1B", "NFKBIA", "CXCL8", "CCL2", "PTGS2"),
    "feedback_regulation": ("SOCS3", "ATF3", "TNFAIP3", "DUSP1", "IL10"),
    "antigen_presentation": (
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "CIITA", "HLA-DRA", "HLA-DRB1",
        "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "CD74",
    ),
}


def paths(root: Path) -> dict[str, Path]:
    out = {
        "published": root / "data" / "published",
        "processed": root / "data" / "processed",
        "reference": root / "data" / "reference",
        "manifests": root / "data" / "manifests",
        "tables": root / "results" / "tables",
        "figures": root / "results" / "figures",
        "reports": root / "results" / "reports",
        "docs": root / "docs",
    }
    for key in ("processed", "manifests", "tables", "figures", "reports", "docs"):
        out[key].mkdir(parents=True, exist_ok=True)
    return out


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    p = paths(root)
    metadata = pd.read_csv(p["published"] / "RNA_sample_metadata.txt.gz", sep="\t")
    counts = pd.read_csv(p["published"] / "RNA_count_matrix.txt.gz", sep="\t", index_col=0)
    genes = pd.read_csv(p["published"] / "RNA_gene_metadata.txt.gz", sep="\t").set_index("gene_id")
    primary = pd.read_csv(p["tables"] / "interaction_gene_results.csv").set_index("gene_id")
    comparison = pd.read_csv(p["tables"] / "interaction_model_comparison.csv").set_index("gene_id")
    normalization = pd.read_csv(p["tables"] / "normalization_sensitivity.csv").set_index("gene_id")
    contrasts = pd.read_csv(
        p["processed"] / "phase4_donor_interaction_contrasts.tsv.gz", sep="\t", index_col=0
    )
    size_factors = pd.read_csv(p["tables"] / "median_ratio_size_factors.csv")
    if list(counts.columns) != list(metadata.sample_id):
        raise ValueError("Published counts and metadata are not identically ordered")
    if not primary.index.equals(comparison.index):
        raise ValueError("Phase 4 primary and count-comparison tables are not aligned")
    if set(primary.index) != set(contrasts.index):
        raise ValueError("Saved donor contrasts do not match Phase 4 tested genes")
    return {
        "metadata": metadata,
        "counts": counts,
        "genes": genes,
        "primary": primary,
        "comparison": comparison,
        "normalization": normalization,
        "contrasts": contrasts.loc[primary.index],
        "size_factors": size_factors,
    }


def reliable_mask(comparison: pd.DataFrame, effect_threshold: float = 0.5) -> pd.Series:
    """Pre-specified cross-model, material-effect validation rule."""
    return (
        comparison["count_reliable"].astype(bool)
        & comparison["same_direction"].astype(bool)
        & (comparison["course_adjusted_p_value"] < FDR)
        & (comparison["count_adjusted_p_value"] < FDR)
        & (comparison["course_interaction_log2"].abs() >= effect_threshold)
    )


def paired_condition_arrays(
    frame: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[list[str], dict[str, np.ndarray]]:
    donors = sorted(metadata.donor.unique())
    lookup = metadata.set_index(["donor", "condition"])["sample_id"]
    arrays = {
        c: frame.loc[:, [lookup.loc[(d, c)] for d in donors]].to_numpy(dtype=float)
        for c in "ABCD"
    }
    return donors, arrays


def calibrate_interaction_null(
    contrasts: pd.DataFrame, n_permutations: int = 500, seed: int = 607
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Sign-flip centered interaction components while preserving factorial main effects.

    In the balanced A/B/C/D design, the interaction basis vector (+1,-1,-1,+1)
    is orthogonal to donor, IFN-gamma and Salmonella main-effect basis vectors.
    Centering removes the fitted interaction and sign-flipping only this component
    therefore constructs a paired null without permuting treatment labels.
    """
    values = contrasts.to_numpy(dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)
    n = values.shape[1]
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, n_permutations))
    means = values @ signs / n
    sumsq = np.square(values).sum(axis=1, keepdims=True)
    variances = np.maximum((sumsq - n * np.square(means)) / (n - 1), 0)
    ses = np.sqrt(variances / n)
    t_values = np.divide(means, ses, out=np.zeros_like(means), where=ses > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df=n - 1)
    rows: list[dict[str, float | int]] = []
    for i in range(n_permutations):
        p = p_values[:, i]
        q = np.asarray(benjamini_hochberg(pd.Series(p)), dtype=float)
        rows.append({
            "permutation": i + 1,
            "nominal_p_lt_0_05_fraction": float(np.mean(p < .05)),
            "bh_fdr_lt_0_05_count": int(np.sum(q < .05)),
            "median_t": float(np.median(t_values[:, i])),
        })
    frame = pd.DataFrame(rows)
    summary: dict[str, float | int | str] = {
        "method": "centered donor-interaction sign flips; common donor signs across genes",
        "seed": seed,
        "n_permutations": n_permutations,
        "n_genes": values.shape[0],
        "n_donors": n,
        "mean_nominal_p_lt_0_05_fraction": float(frame.nominal_p_lt_0_05_fraction.mean()),
        "p95_nominal_p_lt_0_05_fraction": float(frame.nominal_p_lt_0_05_fraction.quantile(.95)),
        "median_bh_discoveries": float(frame.bh_fdr_lt_0_05_count.median()),
        "mean_bh_discoveries": float(frame.bh_fdr_lt_0_05_count.mean()),
        "p95_bh_discoveries": float(frame.bh_fdr_lt_0_05_count.quantile(.95)),
        "max_bh_discoveries": int(frame.bh_fdr_lt_0_05_count.max()),
        "permutation_fraction_with_any_bh_discovery": float(
            np.mean(frame.bh_fdr_lt_0_05_count > 0)
        ),
    }
    return frame, summary


def _fasta_gene_sequences(path: Path, wanted: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    gene: str | None = None
    chunks: list[str] = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if gene in wanted and chunks:
                    result[gene].append("".join(chunks).upper())
                match = re.search(r"\bgene:(ENSG\d+)", line)
                gene = match.group(1) if match else None
                chunks = []
            else:
                chunks.append(line.strip())
        if gene in wanted and chunks:
            result[gene].append("".join(chunks).upper())
    return dict(result)


def _kmers(sequences: list[str], k: int) -> set[str]:
    return {seq[i:i + k] for seq in sequences for i in range(max(0, len(seq) - k + 1))}


def ifnl_sequence_ambiguity(reference: Path, symbol_to_id: dict[str, str]) -> pd.DataFrame:
    ids = set(symbol_to_id.values())
    sequences = _fasta_gene_sequences(reference, ids)
    rows = []
    for symbol, gene_id in symbol_to_id.items():
        row: dict[str, float | int | str] = {
            "gene_name": symbol,
            "gene_id": gene_id,
            "n_ensembl79_cdna_transcripts": len(sequences.get(gene_id, [])),
        }
        for k in (31, 75, 100):
            own = _kmers(sequences.get(gene_id, []), k)
            other = set().union(*[
                _kmers(sequences.get(other_id, []), k)
                for other_id in ids if other_id != gene_id
            ])
            row[f"shared_exact_{k}mer_fraction"] = len(own & other) / len(own) if own else np.nan
            row[f"n_exact_{k}mers"] = len(own)
        rows.append(row)
    return pd.DataFrame(rows).set_index("gene_id")


def validate_ifnl(
    inputs: dict[str, pd.DataFrame], root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = inputs["counts"]
    metadata = inputs["metadata"]
    genes = inputs["genes"]
    primary = inputs["primary"]
    comparison = inputs["comparison"]
    normalization = inputs["normalization"]
    hits = genes.loc[genes.gene_name.isin(IFNL_SYMBOLS)].copy()
    if set(hits.gene_name) != set(IFNL_SYMBOLS) or hits.gene_name.duplicated().any():
        raise ValueError("IFNL1-3 must each map to one unambiguous Ensembl gene")
    symbol_to_id = dict(zip(hits.gene_name, hits.index))
    seq = ifnl_sequence_ambiguity(
        paths(root)["reference"] / "Homo_sapiens.GRCh38.cdna.all.release79.fa.gz",
        symbol_to_id,
    )
    size_factors = inputs["size_factors"].set_index("sample_id").loc[metadata.sample_id, "size_factor"]
    normalized = counts.div(size_factors, axis=1)
    donors, raw_arrays = paired_condition_arrays(counts.loc[list(symbol_to_id.values())], metadata)
    _, norm_arrays = paired_condition_arrays(normalized.loc[list(symbol_to_id.values())], metadata)
    rows = []
    donor_rows = []
    for symbol in IFNL_SYMBOLS:
        gene_id = symbol_to_id[symbol]
        idx = list(symbol_to_id.values()).index(gene_id)
        row: dict[str, object] = {
            "gene_id": gene_id,
            "gene_name": symbol,
            "gene_biotype": hits.loc[gene_id, "gene_biotype"],
            "chr": str(hits.loc[gene_id, "chr"]),
            "start": int(hits.loc[gene_id, "start"]),
            "end": int(hits.loc[gene_id, "end"]),
            "phase4_filter_count_ge_10_n_samples": int((counts.loc[gene_id] >= 10).sum()),
            "raw_detected_gt_0_fraction_all": float((counts.loc[gene_id] > 0).mean()),
            "primary_interaction_log2_pc_0_5": float(primary.loc[gene_id, "mean_interaction_log2"]),
            "primary_adjusted_p_value": float(primary.loc[gene_id, "adjusted_p_value"]),
            "count_interaction_log2": float(comparison.loc[gene_id, "count_interaction_log2"]),
            "count_adjusted_p_value": float(comparison.loc[gene_id, "count_adjusted_p_value"]),
            "count_reliable": bool(comparison.loc[gene_id, "count_reliable"]),
            "cqn_interaction": float(normalization.loc[gene_id, "cqn_interaction"]),
            "cqn_adjusted_p_value": float(normalization.loc[gene_id, "cqn_adjusted_p_value"]),
        }
        for c in "ABCD":
            values = raw_arrays[c][idx]
            row[f"raw_mean_{c}"] = float(values.mean())
            row[f"raw_median_{c}"] = float(np.median(values))
            row[f"detected_gt_0_fraction_{c}"] = float(np.mean(values > 0))
            row[f"count_ge_10_fraction_{c}"] = float(np.mean(values >= 10))
        for pc in (.1, .5, 1.0, 5.0):
            log_arrays = {c: np.log2(norm_arrays[c][idx] + pc) for c in "ABCD"}
            donor_effect = log_arrays["D"] - log_arrays["B"] - log_arrays["C"] + log_arrays["A"]
            row[f"interaction_log2_pseudocount_{pc:g}"] = float(donor_effect.mean())
            row[f"p_value_pseudocount_{pc:g}"] = float(stats.ttest_1samp(donor_effect, 0).pvalue)
            if pc == .5:
                for donor, a, b, c, d, effect in zip(
                    donors, log_arrays["A"], log_arrays["B"], log_arrays["C"],
                    log_arrays["D"], donor_effect
                ):
                    donor_rows.append({
                        "gene_id": gene_id, "gene_name": symbol, "donor": donor,
                        "log2_A": a, "log2_B": b, "log2_C": c, "log2_D": d,
                        "interaction_log2": effect,
                    })
                row["positive_donor_fraction"] = float(np.mean(donor_effect > 0))
                row["D_highest_condition_donor_fraction"] = float(np.mean(
                    (log_arrays["D"] >= log_arrays["A"])
                    & (log_arrays["D"] >= log_arrays["B"])
                    & (log_arrays["D"] >= log_arrays["C"])
                ))
        row.update(seq.loc[gene_id].to_dict())
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(donor_rows)


def negative_interpretation(
    inputs: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    genes = inputs["genes"]
    selected = genes.loc[genes.gene_name.isin(NEGATIVE_SYMBOLS)]
    if selected.gene_name.duplicated().any() or set(selected.gene_name) != set(NEGATIVE_SYMBOLS):
        raise ValueError("Negative-interaction panel is not uniquely annotated")
    primary = inputs["primary"]
    comparison = inputs["comparison"]
    size_factors = inputs["size_factors"].set_index("sample_id").loc[
        inputs["metadata"].sample_id, "size_factor"
    ]
    normalized_log = np.log2(inputs["counts"].div(size_factors, axis=1) + .5)
    donors, arrays = paired_condition_arrays(normalized_log.loc[selected.index], inputs["metadata"])
    rows = []
    for gene_id, annotation in selected.iterrows():
        idx = list(selected.index).index(gene_id)
        a, b, c, d = (arrays[k][idx] for k in "ABCD")
        mean_a, mean_b, mean_c, mean_d = (float(x.mean()) for x in (a, b, c, d))
        if mean_d < min(mean_b, mean_c):
            category = "combined below both single treatments"
        elif mean_d < mean_b:
            category = "Salmonella lowers expression after priming"
        elif mean_d < mean_c:
            category = "priming lowers expression during infection"
        else:
            category = "sub-additive without combined-condition suppression"
        rows.append({
            "gene_id": gene_id,
            "gene_name": annotation.gene_name,
            "interaction_log2": primary.loc[gene_id, "mean_interaction_log2"],
            "adjusted_p_value": primary.loc[gene_id, "adjusted_p_value"],
            "count_reliable": bool(comparison.loc[gene_id, "count_reliable"]),
            "mean_A": mean_a, "mean_B": mean_b, "mean_C": mean_c, "mean_D": mean_d,
            "IFNg_effect_uninfected_B_minus_A": mean_b - mean_a,
            "IFNg_effect_infected_D_minus_C": mean_d - mean_c,
            "Salmonella_effect_unprimed_C_minus_A": mean_c - mean_a,
            "Salmonella_effect_primed_D_minus_B": mean_d - mean_b,
            "donor_fraction_D_lt_B": float(np.mean(d < b)),
            "donor_fraction_D_lt_C": float(np.mean(d < c)),
            "donor_fraction_D_lt_both": float(np.mean((d < b) & (d < c))),
            "interpretation": category,
        })
    return pd.DataFrame(rows).sort_values("interaction_log2")


def gene_reliability(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparison = inputs["comparison"].copy()
    normalization = inputs["normalization"]
    counts = inputs["counts"].loc[comparison.index]
    robust = reliable_mask(comparison)
    out = comparison.copy()
    out["raw_mean_count"] = counts.mean(axis=1)
    out["raw_median_count"] = counts.median(axis=1)
    out["raw_detected_fraction"] = (counts > 0).mean(axis=1)
    out["count_ge_10_fraction"] = (counts >= 10).mean(axis=1)
    out["well_expressed_half_samples"] = (
        (out.raw_median_count >= 10) & (out.raw_detected_fraction >= .5)
    )
    out["cqn_interaction"] = normalization.loc[out.index, "cqn_interaction"]
    out["cqn_adjusted_p_value"] = normalization.loc[out.index, "cqn_adjusted_p_value"]
    out["cqn_same_direction"] = (
        np.sign(out.course_interaction_log2) == np.sign(out.cqn_interaction)
    )
    out["robust_material_interaction"] = robust
    reverse_program = {
        symbol: program for program, symbols in PROGRAMS.items() for symbol in symbols
    }
    out["phase5_program"] = out.gene_name.map(reverse_program)
    return out.reset_index()


def program_summary(reliability: pd.DataFrame) -> pd.DataFrame:
    panel = reliability.loc[reliability.phase5_program.notna()].copy()
    panel["direction"] = np.where(panel.course_interaction_log2 > 0, "positive", "negative")
    rows = []
    for program, frame in panel.groupby("phase5_program", sort=True):
        eligible = frame.loc[frame.well_expressed_half_samples & frame.count_reliable]
        robust = eligible.loc[eligible.robust_material_interaction]
        rows.append({
            "program": program,
            "n_prespecified": len(PROGRAMS[program]),
            "n_found": len(frame),
            "n_well_expressed_and_count_reliable": len(eligible),
            "n_robust_material": len(robust),
            "n_robust_positive": int((robust.course_interaction_log2 > 0).sum()),
            "n_robust_negative": int((robust.course_interaction_log2 < 0).sum()),
            "median_interaction_eligible": float(eligible.course_interaction_log2.median()) if len(eligible) else np.nan,
            "genes_excluded": ";".join(frame.loc[~(frame.well_expressed_half_samples & frame.count_reliable), "gene_name"]),
        })
    return pd.DataFrame(rows)


def parse_obo(path: Path) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    names: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    namespace: dict[str, str] = {}
    current: dict[str, object] = {}

    def commit(term: dict[str, object]) -> None:
        term_id = term.get("id")
        if isinstance(term_id, str) and not term.get("obsolete", False):
            names[term_id] = str(term.get("name", ""))
            namespace[term_id] = str(term.get("namespace", ""))
            parents[term_id].update(term.get("parents", set()))

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit(current)
                current = {"parents": set()}
            elif line.startswith("["):
                commit(current)
                current = {}
            elif current:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:]
                elif line.startswith("is_a: "):
                    current["parents"].add(line.split()[1])
                elif line.startswith("relationship: part_of "):
                    current["parents"].add(line.split()[2])
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
    commit(current)
    return names, dict(parents), namespace


def go_symbol_annotations(
    gaf_path: Path, parents: dict[str, set[str]], namespace: dict[str, str]
) -> dict[str, set[str]]:
    direct: dict[str, set[str]] = defaultdict(set)
    with gzip.open(gaf_path, "rt") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 13:
                continue
            symbol, qualifier, go_id, aspect, taxon = fields[2], fields[3], fields[4], fields[8], fields[12]
            if "NOT" in qualifier.split("|") or aspect != "P" or "taxon:9606" not in taxon:
                continue
            if namespace.get(go_id) == "biological_process":
                direct[symbol].add(go_id)

    cache: dict[str, set[str]] = {}

    def ancestors(term: str, active: set[str] | None = None) -> set[str]:
        if term in cache:
            return cache[term]
        active = set() if active is None else active
        if term in active:
            return {term}
        result = {term}
        for parent in parents.get(term, set()):
            result.update(ancestors(parent, active | {term}))
        cache[term] = result
        return result

    return {symbol: set().union(*(ancestors(term) for term in terms)) for symbol, terms in direct.items()}


def local_go_enrichment(
    reliability: pd.DataFrame, obo_path: Path, gaf_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    names, parents, namespace = parse_obo(obo_path)
    annotations = go_symbol_annotations(gaf_path, parents, namespace)
    eligible = reliability.loc[
        reliability.count_reliable
        & reliability.gene_name.notna()
        & ~reliability.gene_name.duplicated(keep=False)
    ].copy()
    universe = set(eligible.gene_name) & set(annotations)
    robust = eligible.loc[eligible.robust_material_interaction]
    sets = {
        "positive": set(robust.loc[robust.course_interaction_log2 > 0, "gene_name"]) & universe,
        "negative": set(robust.loc[robust.course_interaction_log2 < 0, "gene_name"]) & universe,
    }
    term_genes: dict[str, set[str]] = defaultdict(set)
    for symbol in universe:
        for term in annotations[symbol]:
            term_genes[term].add(symbol)
    rows = []
    for direction, selected in sets.items():
        for term, annotated in term_genes.items():
            k = len(annotated)
            overlap = selected & annotated
            if k < 10 or k > 2000:
                continue
            table = [[len(overlap), len(selected) - len(overlap)],
                     [k - len(overlap), len(universe) - len(selected) - k + len(overlap)]]
            odds, p_value = stats.fisher_exact(table, alternative="greater")
            rows.append({
                "direction": direction,
                "go_id": term,
                "go_term": names.get(term, ""),
                "selected_genes_annotated": len(selected),
                "universe_genes_annotated": len(universe),
                "term_genes_in_universe": k,
                "overlap_count": len(overlap),
                "odds_ratio": odds,
                "p_value": p_value,
                "overlap_genes": ";".join(sorted(overlap)),
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_p_value"] = np.nan
        for direction, indices in result.groupby("direction").groups.items():
            result.loc[indices, "adjusted_p_value"] = np.asarray(
                benjamini_hochberg(result.loc[indices, "p_value"]), dtype=float
            )
        result = result.sort_values(["direction", "adjusted_p_value", "p_value"])
    audit = {
        "ontology_terms": len(names),
        "human_symbols_with_bp_annotations": len(annotations),
        "eligible_unique_reliable_tested_symbols": len(eligible),
        "annotated_universe_symbols": len(universe),
        "selected_positive_symbols": len(sets["positive"]),
        "selected_negative_symbols": len(sets["negative"]),
        "term_size_limits": [10, 2000],
        "terms_tested_by_direction": result.groupby("direction").size().to_dict() if not result.empty else {},
        "significant_terms_by_direction": (
            result.loc[result.adjusted_p_value < FDR].groupby("direction").size().to_dict()
            if not result.empty else {}
        ),
        "multiple_testing": "BH separately within positive and negative tests",
        "annotation_propagation": "is_a and part_of ancestors",
    }
    return result, audit


def normalization_offset(inputs: dict[str, pd.DataFrame]) -> dict[str, float | int]:
    primary = inputs["primary"]
    normalization = inputs["normalization"]
    metadata = inputs["metadata"]
    sf = inputs["size_factors"].set_index("sample_id").loc[metadata.sample_id, "size_factor"]
    sf_frame = pd.DataFrame({"donor": metadata.donor, "condition": metadata.condition, "sf": sf.to_numpy()})
    pivot = sf_frame.pivot(index="donor", columns="condition", values="sf")
    sf_interaction = np.log2(pivot.D) - np.log2(pivot.B) - np.log2(pivot.C) + np.log2(pivot.A)
    effects = primary.mean_interaction_log2
    cqn = normalization.cqn_interaction
    return {
        "primary_median_effect": float(effects.median()),
        "primary_mean_effect": float(effects.mean()),
        "primary_positive_fraction": float(np.mean(effects > 0)),
        "cqn_median_effect": float(cqn.median()),
        "cqn_mean_effect": float(cqn.mean()),
        "median_size_factor_log2_interaction": float(sf_interaction.median()),
        "mean_size_factor_log2_interaction": float(sf_interaction.mean()),
        "normalization_contribution_sign": "normalized interaction subtracts the size-factor interaction",
    }


def _plot_ifnl_conditions(ifnl: pd.DataFrame, path: Path) -> None:
    long = ifnl.melt(
        id_vars=["gene_name"], value_vars=[f"raw_mean_{c}" for c in "ABCD"],
        var_name="condition", value_name="raw_mean_count"
    )
    long["condition"] = long.condition.str[-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(data=long, x="gene_name", y="raw_mean_count", hue="condition", ax=ax,
                palette={"A": "#6b6b6b", "B": "#2b8cbe", "C": "#e6550d", "D": "#756bb1"})
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="", ylabel="Raw mean count (symlog scale)",
           title="IFNL1-3 raw expression across the four conditions")
    ax.legend(title="Condition", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_ifnl_donors(donor: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (symbol, frame) in zip(axes, donor.groupby("gene_name", sort=True)):
        ax.axhline(0, color="black", lw=.7)
        ax.scatter(np.arange(len(frame)), np.sort(frame.interaction_log2), s=13, alpha=.7,
                   color="#7b3294")
        ax.set_title(symbol)
        ax.set_xlabel("Donors, ordered by interaction")
    axes[0].set_ylabel("Donor interaction (log2 units)")
    fig.suptitle("IFNL interaction effects are donor-level, not pooled-sample artifacts")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_positive_negative(primary: pd.DataFrame, reliability: pd.DataFrame, path: Path) -> None:
    merged = primary.join(reliability.set_index("gene_id")[["robust_material_interaction"]])
    frame = merged.loc[merged.robust_material_interaction].copy()
    frame["direction"] = np.where(frame.mean_interaction_log2 > 0, "Positive", "Negative")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.violinplot(data=frame, x="direction", y="mean_interaction_log2", hue="direction",
                   palette={"Positive": "#b2182b", "Negative": "#2166ac"}, legend=False,
                   inner="quart", cut=0, ax=axes[0])
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set(xlabel="", ylabel="Paired interaction (log2 units)", title="Robust material interactions")
    counts = frame.direction.value_counts().rename_axis("direction").reset_index(name="genes")
    sns.barplot(data=counts, x="direction", y="genes", hue="direction",
                palette={"Positive": "#b2182b", "Negative": "#2166ac"}, legend=False, ax=axes[1])
    axes[1].set(xlabel="", ylabel="Genes", title="Counts after cross-model validation")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_abundance(reliability: pd.DataFrame, path: Path) -> None:
    frame = reliability.copy()
    frame["status"] = np.where(frame.robust_material_interaction, "Robust material", "Other tested")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=frame, x="raw_mean_count", y="course_interaction_log2", hue="status",
                    palette={"Robust material": "#7b3294", "Other tested": "#9e9e9e"},
                    s=13, alpha=.45, linewidth=0, ax=ax)
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="Raw mean count (log scale)", ylabel="Paired interaction (log2 units)",
           title="Interaction magnitude versus expression abundance")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(root: Path, n_permutations: int = 500) -> dict[str, object]:
    p = paths(root)
    inputs = load_inputs(root)
    comparison = inputs["comparison"]
    robust = reliable_mask(comparison)
    reliability = gene_reliability(inputs)
    ifnl, ifnl_donor = validate_ifnl(inputs, root)
    negative = negative_interpretation(inputs)
    programs = program_summary(reliability)
    null_table, null_summary = calibrate_interaction_null(
        inputs["contrasts"], n_permutations=n_permutations
    )
    enrichment, enrichment_audit = local_go_enrichment(
        reliability, p["reference"] / "go-basic.obo", p["reference"] / "goa_human.gaf.gz"
    )
    offset = normalization_offset(inputs)

    reliability.to_csv(p["tables"] / "phase5_gene_reliability.csv", index=False)
    ifnl.to_csv(p["tables"] / "ifnl_gene_validation.csv", index=False)
    ifnl_donor.to_csv(p["tables"] / "ifnl_donor_validation.csv", index=False)
    negative.to_csv(p["tables"] / "negative_interaction_interpretation.csv", index=False)
    programs.to_csv(p["tables"] / "phase5_program_summary.csv", index=False)
    null_table.to_csv(p["tables"] / "phase5_null_calibration.csv", index=False)
    enrichment.to_csv(p["tables"] / "phase5_enrichment_results.csv", index=False)

    reference_manifest = pd.DataFrame([
        {"filename": "go-basic.obo", "source_url": "https://current.geneontology.org/ontology/go-basic.obo"},
        {"filename": "goa_human.gaf.gz", "source_url": "https://current.geneontology.org/annotations/goa_human.gaf.gz"},
        {"filename": "Homo_sapiens.GRCh38.cdna.all.release79.fa.gz", "source_url": "https://ftp.ensembl.org/pub/release-79/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz"},
    ])
    reference_manifest["size_bytes"] = reference_manifest.filename.map(
        lambda name: (p["reference"] / name).stat().st_size
    )
    reference_manifest["sha256"] = reference_manifest.filename.map(
        lambda name: _sha256(p["reference"] / name)
    )
    reference_manifest.to_csv(p["manifests"] / "phase5_reference_manifest.csv", index=False)

    _plot_ifnl_conditions(ifnl, p["figures"] / "ifnl_four_condition_expression.png")
    _plot_ifnl_donors(ifnl_donor, p["figures"] / "ifnl_donor_response.png")
    _plot_positive_negative(inputs["primary"], reliability, p["figures"] / "positive_negative_interaction_comparison.png")
    _plot_abundance(reliability, p["figures"] / "interaction_vs_expression_abundance.png")

    summary: dict[str, object] = {
        "phase4_primary_tested": len(comparison),
        "phase4_primary_fdr_significant": int(comparison.course_significant.sum()),
        "phase4_primary_fdr_fraction": float(comparison.course_significant.mean()),
        "count_unreliable": int((~comparison.count_reliable.astype(bool)).sum()),
        "primary_significant_among_count_reliable": int((comparison.course_significant & comparison.count_reliable).sum()),
        "count_significant_among_count_reliable": int((comparison.count_significant & comparison.count_reliable).sum()),
        "cross_model_significant_same_direction_reliable": int((
            comparison.course_significant & comparison.count_significant
            & comparison.same_direction & comparison.count_reliable
        ).sum()),
        "robust_material_interactions": int(robust.sum()),
        "robust_material_positive": int((robust & (comparison.course_interaction_log2 > 0)).sum()),
        "robust_material_negative": int((robust & (comparison.course_interaction_log2 < 0)).sum()),
        "robust_material_cqn_same_direction_and_significant": int((
            reliability.robust_material_interaction
            & reliability.cqn_same_direction
            & (reliability.cqn_adjusted_p_value < FDR)
        ).sum()),
        "null_calibration": null_summary,
        "normalization_offset": offset,
        "go_enrichment": enrichment_audit,
    }
    (p["tables"] / "phase5_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
