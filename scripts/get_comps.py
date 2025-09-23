#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ---------- Repo-relative defaults ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA = os.path.join(
    PROJECT_ROOT, "output", "assessment_pin", "model_assessment_pin.parquet"
)
DEFAULT_LEAVES = os.path.join(
    PROJECT_ROOT, "output", "intermediate", "pin_leaves.parquet"
)
DEFAULT_OUTDIR = os.path.join(PROJECT_ROOT, "output", "comp_sheets")


# ---------------- Utils ----------------
def jaccard_similarity(sig1: str, sig2: str) -> float:
    if not isinstance(sig1, str) or not isinstance(sig2, str):
        return 0.0
    set1, set2 = set(str(sig1).split("_")), set(str(sig2).split("_"))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _fmt_pin(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return s
    return digits.zfill(14)[:14]


def _fmt_pin_dashed(pin14: str) -> str:
    s = "".join(ch for ch in str(pin14) if ch.isdigit()).zfill(14)[:14]
    return f"{s[0:2]}-{s[2:4]}-{s[4:7]}-{s[7:10]}-{s[10:14]}"


def fmt_int_commas(x):
    v = pd.to_numeric(x, errors="coerce")
    return v.apply(lambda n: "" if pd.isna(n) else f"{int(round(n)):,}")


def fmt_money0(x):
    v = pd.to_numeric(x, errors="coerce")
    return v.apply(lambda n: "" if pd.isna(n) else f"${int(round(n)):,}")


def fmt_2(x):
    v = pd.to_numeric(x, errors="coerce")
    return v.apply(lambda n: "" if pd.isna(n) else f"{n:.2f}")


def _guess_ccao_pin_column(df: pd.DataFrame) -> str | None:
    """Find a column in the CCAO CSV that contains PINs and return its name."""
    cand_names = [
        "PIN",
        "pin",
        "Pin",
        "Comp PIN",
        "comp_pin",
        "COMP_PIN",
        "CCAO_PIN",
        "ccao_pin",
        "Ccao Pin",
        "PIN_DASHED",
        "pin_dashed",
        "Pin-Dashed",
    ]
    for name in cand_names:
        if name in df.columns:
            return name
    # heuristic: first column containing many digits/dashes/length ~14–17
    for c in df.columns:
        sample = df[c].astype(str).head(20).str.replace(r"[^0-9-]", "", regex=True)
        if (sample.str.len().median() >= 10) and (
            sample.str.contains(r"\d", regex=True).mean() > 0.8
        ):
            return c
    return None


# --------- column map (edit if your schema differs) ----------
COLMAP = {
    "pin": "meta_pin",
    "addr": "loc_property_address",
    "total_av": "pred_pin_final_fmv_round",
    "fmv": "pred_pin_final_fmv",
    "bldg_av": "pred_pin_final_fmv_bldg",
    "land_av": "pred_pin_final_fmv_land",
    "sqft": "char_total_bldg_sf",
}

# Candidate sale columns (ordered by preference)
SALE_PRICE_CANDS = ["sale_recent_1_price", "sale_ratio_study_price"]
SALE_DATE_CANDS = ["sale_recent_1_date", "sale_ratio_study_date"]

# (We’re no longer outputting deed/outlier, so no need to chase those here.)

NUMERIC_KEEP = [
    "char_total_bldg_sf",
    "char_land_sf",
    "land_rate_per_sqft",
    "prior_far_tot",
    "prior_near_tot",
    "pred_pin_final_fmv_round_no_prorate",
    "pred_pin_final_fmv_bldg_no_prorate",
    "pred_pin_final_fmv_land",
    "pred_pin_final_fmv_bldg",
]

META_COLS = ["meta_pin", "meta_township_code", "meta_nbhd_code", "loc_property_address"]
VALUE_COLS = [
    COLMAP["total_av"],
    COLMAP["fmv"],
    COLMAP["bldg_av"],
    COLMAP["land_av"],
    COLMAP["sqft"],
]
READ_COLS = sorted(
    set(NUMERIC_KEEP + META_COLS + VALUE_COLS + SALE_PRICE_CANDS + SALE_DATE_CANDS)
)


# ---------------- Loaders ----------------
def load_main_table(path_parquet: str) -> pd.DataFrame:
    print(f"loading data from {path_parquet} ...")
    base = pd.read_parquet(path_parquet, engine="pyarrow")
    cols = [c for c in READ_COLS if c in base.columns]
    df = base[cols].copy()

    # Rename core display columns
    rename_map = {
        COLMAP["pin"]: "PIN",
        COLMAP["addr"]: "Property Address",
        COLMAP["total_av"]: "total_assessed_value",
        COLMAP["fmv"]: "FMV",
        COLMAP["bldg_av"]: "Building AV",
        COLMAP["land_av"]: "Land AV",
        COLMAP["sqft"]: "SQFT",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required display columns exist
    for col in [
        "Property Address",
        "FMV",
        "Building AV",
        "Land AV",
        "SQFT",
        "total_assessed_value",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Add sale fields by best-available column
    price_col = next((c for c in SALE_PRICE_CANDS if c in df.columns), None)
    date_col = next((c for c in SALE_DATE_CANDS if c in df.columns), None)
    df["Sale Price"] = df[price_col] if price_col else np.nan
    df["Sale Date"] = (
        pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    )

    # Normalize PIN
    if "PIN" not in df.columns:
        raise SystemExit("❌ Could not find meta_pin to build PIN column.")
    df["PIN"] = df["PIN"].apply(_fmt_pin).astype(str)

    keep = [
        "PIN",
        "Property Address",
        "meta_township_code",
        "meta_nbhd_code",
        "total_assessed_value",
        "FMV",
        "Building AV",
        "Land AV",
        "SQFT",
        "Sale Price",
        "Sale Date",
    ] + [c for c in NUMERIC_KEEP if c in df.columns]
    return df[keep]


def find_leaves_path(user_path: str | None) -> str | None:
    if user_path and os.path.exists(user_path):
        return user_path
    patterns = [
        os.path.join(PROJECT_ROOT, "output", "intermediate", "pin_leaves.parquet"),
        "output/intermediate/pin_leaves.parquet",
        "output/leaves/leaves.parquet",
        "output/leaves/*.parquet",
        "output/**/leaves*.parquet",
    ]
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p):
                return p
    return None


def load_leaves(leaves_path: str | None) -> pd.DataFrame | None:
    path = find_leaves_path(leaves_path)
    if not path:
        print("⚠️  no leaves file found; continuing without leaf values")
        return None
    print(f"loading leaves from {path} ...")
    leaves = pd.read_parquet(path, engine="pyarrow")

    # find pin col
    cols_lower = {c.lower(): c for c in leaves.columns}
    pin_col = cols_lower.get("meta_pin") or cols_lower.get("pin")
    if not pin_col:
        print("⚠️  leaves file missing PIN column; skipping leaf merge")
        return None

    # normalize leaf column name
    if "leaf_value" not in leaves.columns:
        if "leaf_signature" in leaves.columns:
            leaves = leaves.rename(columns={"leaf_signature": "leaf_value"})
        else:
            print(
                "⚠️  leaves file missing leaf_value/leaf_signature; skipping leaf merge"
            )
            return None

    leaves = leaves.rename(columns={pin_col: "PIN"})
    leaves["PIN"] = leaves["PIN"].apply(_fmt_pin).astype(str)
    return leaves[["PIN", "leaf_value"]]


# ---------------- Logic ----------------
def pick_scope(df: pd.DataFrame, subject_row: pd.Series, scope: str) -> pd.DataFrame:
    if scope == "all":
        return df
    if scope == "township" and "meta_township_code" in df.columns:
        return df[df["meta_township_code"] == subject_row["meta_township_code"]]
    if scope == "nbhd" and "meta_nbhd_code" in df.columns:
        return df[df["meta_nbhd_code"] == subject_row["meta_nbhd_code"]]
    return df


def safe_numeric_block(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    num = df[cols].copy()
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0.0).infer_objects(copy=False)
    scaler = StandardScaler()
    return scaler.fit_transform(num)


def neighbors_for_subject(subject_vec: np.ndarray, X: np.ndarray, k: int):
    """
    Return the indexes and cosine-similarity scores of the k nearest neighbors
    to the subject vector within matrix X (excluding the subject itself).
    """
    # be safe if k > population-1
    k_eff = max(1, min(k + 1, X.shape[0]))  # +1 so we can drop the subject row
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k_eff)
    nn.fit(X)
    dists, idxs = nn.kneighbors(subject_vec.reshape(1, -1), return_distance=True)
    d = dists[0]
    i = idxs[0]
    # drop the subject itself (first position)
    return i[1:], (1.0 - d[1:])


def score_pool(
    df_scope: pd.DataFrame,
    subj: pd.Series,
    feats: list[str],
    weights: tuple[float, float],
):
    """Compute similarity, leaf jaccard (and its [0,1] normalization), and composite for ALL rows in df_scope."""
    # similarity backbone (cosine to subject)
    X = safe_numeric_block(df_scope, feats)
    subj_vec = X[df_scope.index.get_indexer([subj.name])[0]]
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=X.shape[0])
    nn.fit(X)
    dists, idxs = nn.kneighbors(subj_vec.reshape(1, -1), return_distance=True)
    sims = 1.0 - dists[0]
    df_scored = df_scope.copy()
    df_scored["similarity"] = sims

    # leaf jaccard (raw) + normalization
    if "leaf_value" in df_scored.columns and pd.notna(subj.get("leaf_value", None)):
        subj_sig = subj["leaf_value"]
        df_scored["leaf_value"] = df_scored["leaf_value"].apply(
            lambda s: jaccard_similarity(subj_sig, s)
        )
        lv = pd.to_numeric(df_scored["leaf_value"], errors="coerce")
        lv_min, lv_max = lv.min(skipna=True), lv.max(skipna=True)
        if pd.notna(lv_min) and pd.notna(lv_max) and lv_max > lv_min:
            lv_norm = (lv - lv_min) / (lv_max - lv_min)
        else:
            lv_norm = pd.Series(0.5, index=df_scored.index)
    else:
        df_scored["leaf_value"] = np.nan
        lv_norm = pd.Series(0.5, index=df_scored.index)

    w_sim, w_leaf = weights
    df_scored["composite"] = (w_sim * df_scored["similarity"]) + (w_leaf * lv_norm)
    # keep a 5-decimal string presentation for export
    for c in ["similarity", "leaf_value", "composite"]:
        df_scored[c] = pd.to_numeric(df_scored[c], errors="coerce").round(5)

    return df_scored


def compute_and_write(
    df_scope_filtered,
    df_scope_unfiltered,
    subj,
    feats_filtered,
    feats_unfiltered,
    subject_pin,
    outdir,
    k,
    weights,
    suffix,
    ccao_df=None,
):
    """Write top-k from the FILTERED pool, but compare CCAO comps against the
    UNFILTERED (same-scope) universe so you don’t lose matches to date filters."""
    w_sim, w_leaf = weights

    # ---------- score FILTERED pool (used for top-k output) ----------
    X_f = safe_numeric_block(df_scope_filtered, feats_filtered)
    subj_vec_f = X_f[df_scope_filtered.index.get_indexer([subj.name])[0]]
    idxs_f, sims_f = neighbors_for_subject(subj_vec_f, X_f, k)
    comps = df_scope_filtered.iloc[idxs_f].copy()
    comps["similarity"] = sims_f

    # FMV/BAV per sqft (numeric)
    comps["FMV per SQFT"] = np.where(
        pd.to_numeric(comps["SQFT"], errors="coerce") > 0,
        pd.to_numeric(comps["total_assessed_value"], errors="coerce")
        / pd.to_numeric(comps["SQFT"], errors="coerce"),
        np.nan,
    )
    if "Building AV" in comps.columns:
        comps["BAV per SQFT"] = np.where(
            pd.to_numeric(comps["SQFT"], errors="coerce") > 0,
            pd.to_numeric(comps["Building AV"], errors="coerce")
            / pd.to_numeric(comps["SQFT"], errors="coerce"),
            np.nan,
        )
    else:
        comps["BAV per SQFT"] = np.nan

    # leaf similarity (filtered pool uses subject’s leaf too)
    if "leaf_value" in comps.columns and pd.notna(subj.get("leaf_value", None)):
        subj_sig = subj["leaf_value"]
        comps["leaf_value"] = comps["leaf_value"].apply(
            lambda s: jaccard_similarity(subj_sig, s)
        )
        lv = comps["leaf_value"].astype(float)
        lv_min, lv_max = lv.min(skipna=True), lv.max(skipna=True)
        lv_norm = (
            (lv - lv_min) / (lv_max - lv_min)
            if (pd.notna(lv_min) and pd.notna(lv_max) and lv_max > lv_min)
            else pd.Series(0.5, index=comps.index)
        )
    else:
        comps["leaf_value"] = np.nan
        lv_norm = pd.Series(0.5, index=comps.index)

    comps["composite"] = (w_sim * comps["similarity"]) + (w_leaf * lv_norm)

    # numeric coercions
    for c in ["FMV", "Building AV", "Land AV", "total_assessed_value", "SQFT"]:
        if c in comps.columns:
            comps[c] = pd.to_numeric(comps[c], errors="coerce")

    # 5-decimal string motif for scores
    for c in ["similarity", "leaf_value", "composite"]:
        if c in comps.columns:
            comps[c] = pd.to_numeric(comps[c], errors="coerce").round(5)

    # normalized PIN (raw 14) for join/format
    comps["PIN"] = (
        comps["PIN"]
        .astype(str)
        .apply(lambda s: "".join(ch for ch in s if ch.isdigit()).zfill(14)[:14])
    )

    # rename total_assessed_value for presentation
    if "total_assessed_value" in comps.columns:
        comps = comps.rename(columns={"total_assessed_value": "Total AV"})

    # choose & order columns
    cols = [
        "PIN",
        "Property Address",
        "Building AV",
        "Land AV",
        "Total AV",
        "FMV",
        "FMV per SQFT",
        "BAV per SQFT",
        "SQFT",
        "Sale Price",
        "Sale Date",
        # intentionally excluding Deed Type / Outlier Reason from output
        "leaf_value",
        "similarity",
        "composite",
    ]
    out_cols = [c for c in cols if c in comps.columns]
    df_out = comps.sort_values("composite", ascending=False).head(k)[out_cols].copy()

    # fixed 5-decimal string for scores (keeps 1.00000)
    for c in ["leaf_value", "similarity", "composite"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").apply(
                lambda x: "" if pd.isna(x) else f"{x:.5f}"
            )

    # Format for CSV
    df_out["PIN"] = df_out["PIN"].apply(_fmt_pin_dashed)
    for c in ["Building AV", "Land AV", "Total AV", "SQFT", "FMV"]:
        if c in df_out.columns:
            df_out[c] = fmt_int_commas(df_out[c])
    if "Sale Price" in df_out.columns:
        df_out["Sale Price"] = fmt_money0(df_out["Sale Price"])
    if "FMV per SQFT" in df_out.columns:
        df_out["FMV per SQFT"] = fmt_2(df_out["FMV per SQFT"])
    if "BAV per SQFT" in df_out.columns:
        df_out["BAV per SQFT"] = fmt_2(df_out["BAV per SQFT"])
    if "Sale Date" in df_out.columns:
        df_out["Sale Date"] = pd.to_datetime(
            df_out["Sale Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    # write CSV for top-k
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{subject_pin}_comps{suffix}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"✅ wrote {len(df_out)} comps to {out_path}")

    # echo a preview to terminal
    print("\n▶ preview rows:")
    print(df_out.to_dict(orient="records"))

    # ---------- CCAO compare against UNFILTERED universe ----------
    # ---------- CCAO compare against UNFILTERED universe ----------
    if ccao_df is not None and not ccao_df.empty:
        try:
            # Score universe (unfiltered within the same scope)
            feats_u = [c for c in feats_unfiltered if c in df_scope_unfiltered.columns]
            if not feats_u:
                print(
                    "⚠️  No numeric features available in unfiltered universe; skipping CCAO compare."
                )
                return

            X_u = safe_numeric_block(df_scope_unfiltered, feats_u)
            # ensure we can ask for at least 2 neighbors (subject + 1)
            n_u = min(len(df_scope_unfiltered), max(len(df_scope_unfiltered), 2))
            subj_idx_u = df_scope_unfiltered.index.get_indexer([subj.name])[0]
            subj_vec_u = X_u[subj_idx_u]

            nn_u = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_u)
            nn_u.fit(X_u)
            dists_u, idxs_u = nn_u.kneighbors(
                subj_vec_u.reshape(1, -1), return_distance=True
            )
            sims_u = 1.0 - dists_u[0]

            uni = df_scope_unfiltered.iloc[idxs_u[0]].copy()
            uni["similarity"] = sims_u

            # leaf similarity on universe
            if "leaf_value" in uni.columns and pd.notna(subj.get("leaf_value", None)):
                subj_sig = subj["leaf_value"]
                uni["leaf_value"] = uni["leaf_value"].apply(
                    lambda s: jaccard_similarity(subj_sig, s)
                )
                lv = pd.to_numeric(uni["leaf_value"], errors="coerce")
                lv_min, lv_max = lv.min(skipna=True), lv.max(skipna=True)
                if pd.notna(lv_min) and pd.notna(lv_max) and lv_max > lv_min:
                    lv_norm_u = (lv - lv_min) / (lv_max - lv_min)
                else:
                    lv_norm_u = pd.Series(0.5, index=uni.index)
            else:
                uni["leaf_value"] = np.nan
                lv_norm_u = pd.Series(0.5, index=uni.index)

            # composite
            uni["composite"] = (w_sim * uni["similarity"]) + (w_leaf * lv_norm_u)

            # drop the subject if present
            uni = uni.drop(index=[subj.name], errors="ignore")

            # numeric coercions (guard if missing)
            for col in [
                "FMV",
                "Building AV",
                "Land AV",
                "total_assessed_value",
                "SQFT",
                "Sale Price",
            ]:
                if col in uni.columns:
                    uni[col] = pd.to_numeric(uni[col], errors="coerce")

            # per-sqft metrics (Total AV is our FMV stand-in per prior runs)
            if "total_assessed_value" in uni.columns and "SQFT" in uni.columns:
                uni["FMV per SQFT"] = np.where(
                    uni["SQFT"] > 0, uni["total_assessed_value"] / uni["SQFT"], np.nan
                )
            else:
                uni["FMV per SQFT"] = np.nan

            if "Building AV" in uni.columns and "SQFT" in uni.columns:
                uni["BAV per SQFT"] = np.where(
                    uni["SQFT"] > 0, uni["Building AV"] / uni["SQFT"], np.nan
                )
            else:
                uni["BAV per SQFT"] = np.nan

            # prepare join keys + ranking
            uni["PIN14"] = uni["PIN"].astype(str).apply(_fmt_pin)
            # rank across the full in-scope universe
            uni["rank"] = (
                uni["composite"].rank(method="min", ascending=False).astype(int)
            )

            # slim to fields we want in the compare CSV
            uni_small_cols = [
                "PIN14",
                "Property Address",
                "Building AV",
                "Land AV",
                "total_assessed_value",
                "FMV",
                "FMV per SQFT",
                "BAV per SQFT",
                "SQFT",
                "similarity",
                "leaf_value",
                "composite",
                "rank",
            ]
            uni_small = uni[[c for c in uni_small_cols if c in uni.columns]].copy()

            # normalize CCAO CSV pins -> PIN14
            key = _guess_ccao_pin_column(ccao_df) or "PIN"
            if key not in ccao_df.columns:
                print(
                    f"⚠️  Could not find column '{key}' in CCAO CSV; skipping compare."
                )
                return

            cc = ccao_df.copy()
            cc["PIN14"] = cc[key].astype(str).apply(_fmt_pin)
            cc = cc[["PIN14"]].drop_duplicates()

            # merge ccao -> universe
            cmp = cc.merge(uni_small, on="PIN14", how="left", suffixes=("", ""))

            # mark if the CCAO comp also appears in YOUR top-k filtered output
            top_raw = comps.copy()
            top_raw["PIN14"] = top_raw["PIN"].astype(str).apply(_fmt_pin)
            top_raw["topk_pos"] = range(1, len(top_raw) + 1)
            cmp = cmp.merge(top_raw[["PIN14", "topk_pos"]], on="PIN14", how="left")

            # presentation formatting (match your main CSV)
            cmp["PIN"] = cmp["PIN14"].apply(_fmt_pin_dashed)

            # scores to 5 decimals (keep blanks if missing)
            for c in ["leaf_value", "similarity", "composite"]:
                if c in cmp.columns:
                    cmp[c] = pd.to_numeric(cmp[c], errors="coerce").apply(
                        lambda x: "" if pd.isna(x) else f"{x:.5f}"
                    )

            # integer-with-commas (no $)
            for c in ["Building AV", "Land AV", "total_assessed_value", "SQFT", "FMV"]:
                if c in cmp.columns:
                    cmp[c] = fmt_int_commas(cmp[c])

            # per-SQFT as 2 decimals
            for c in ["FMV per SQFT", "BAV per SQFT"]:
                if c in cmp.columns:
                    cmp[c] = fmt_2(cmp[c])

            # rename total_assessed_value -> Total AV for output consistency
            if "total_assessed_value" in cmp.columns:
                cmp = cmp.rename(columns={"total_assessed_value": "Total AV"})

            cmp["in_top_k"] = cmp["topk_pos"].notna()

            keep_cmp = [
                "PIN",
                "Property Address",
                "Building AV",
                "Land AV",
                "Total AV",
                "FMV",
                "FMV per SQFT",
                "BAV per SQFT",
                "SQFT",
                "rank",
                "in_top_k",
                "topk_pos",
                "similarity",
                "leaf_value",
                "composite",
            ]
            keep_cmp = [c for c in keep_cmp if c in cmp.columns]
            cmp_out = cmp[keep_cmp].sort_values(
                ["in_top_k", "rank"], ascending=[False, True]
            )

            compare_path = os.path.join(
                outdir, f"{subject_pin}_ccao_compare{suffix}.csv"
            )
            cmp_out.to_csv(compare_path, index=False)
            print(f"📝 wrote CCAO comparison to {compare_path}")
            print(f"   · universe size: {len(df_scope_unfiltered):,}")
            print(
                f"   · CCAO pins provided: {len(cc):,}, matched in universe: {cmp_out['PIN'].notna().sum():,}"
            )

        except Exception as e:
            # never abort the whole run — just report and continue
            print(f"⚠️  CCAO compare failed gracefully: {e}")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Return top-k comps for a subject PIN.")
    ap.add_argument("pin", help="subject PIN (e.g., 01011000090000)")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument(
        "--leaves",
        default=DEFAULT_LEAVES,
        help="optional leaves parquet path (auto-detected if omitted)",
    )
    ap.add_argument("--scope", choices=["all", "township", "nbhd"], default="township")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument(
        "--weights",
        nargs="+",
        default=["0.70,0.30"],
        help="List of weight pairs sim,leaf (e.g. '0.7,0.3').",
    )
    ap.add_argument(
        "--ccao",
        default=None,
        help="CSV with CCAO-provided comps (one column of PINs; default column name 'PIN').",
    )
    ap.add_argument(
        "--ccao_pin_col", default="PIN", help="Column name in --ccao containing PINs."
    )
    args = ap.parse_args()

    print("Paths in use:")
    print("  data   :", os.path.abspath(args.data))
    print("  leaves :", os.path.abspath(args.leaves) if args.leaves else "(auto)")
    print("  outdir :", os.path.abspath(args.outdir))

    df = load_main_table(args.data)

    subject_pin = _fmt_pin(args.pin)
    if subject_pin not in set(df["PIN"]):
        raise SystemExit(f"❌ PIN {subject_pin} not found in {args.data}")

    # attach leaves (optional)
    leaves = load_leaves(args.leaves)
    if leaves is not None:
        df = df.merge(leaves, on="PIN", how="left")
    else:
        df["leaf_value"] = np.nan

    subj = df.loc[df["PIN"] == subject_pin].iloc[0]
    cand = pick_scope(df, subj, args.scope).copy()

    # ---- restrict comps to sales within 3 years prior to Jan 1, 2025 ----
    cand.loc[:, "Sale Date"] = pd.to_datetime(cand["Sale Date"], errors="coerce")
    cutoff = pd.Timestamp("2022-01-01")  # inclusive: 2022-01-01 through 2024-12-31
    pool = cand[cand["Sale Date"] >= cutoff].copy()
    pool = pool.reindex(columns=cand.columns)
    if subj.name not in pool.index:
        pool = pd.concat(
            [subj.to_frame().T.reindex(columns=pool.columns), pool],
            axis=0,
            ignore_index=False,
        )
    kept = (len(pool) - 1) if subj.name in pool.index else len(pool)
    print(
        f"📅 date-filter kept {kept} candidate rows (>= {cutoff.date()}) within scope='{args.scope}'"
    )

    # features for similarity
    feats = [c for c in NUMERIC_KEEP if c in pool.columns]
    if not feats:
        raise SystemExit(
            "❌ No numeric features available for similarity. Check NUMERIC_KEEP list."
        )

    # optional: read CCAO comps CSV
    ccao_df = None
    if args.ccao:
        try:
            raw_ccao = pd.read_csv(args.ccao)
            if args.ccao_pin_col not in raw_ccao.columns:
                raise SystemExit(
                    f"❌ --ccao_pin_col '{args.ccao_pin_col}' not found in {args.ccao}"
                )
            # normalize into a single-column frame
            ccao_df = raw_ccao[[args.ccao_pin_col]].rename(
                columns={args.ccao_pin_col: "PIN"}
            )
        except Exception as e:
            raise SystemExit(f"❌ Failed to read --ccao file: {e}")

    # run for each weight split -> separate CSVs (default is one: 70/30)
    unique_weights = []
    for w in args.weights:
        try:
            pair = tuple(float(x) for x in w.split(","))
        except Exception:
            raise SystemExit(f"Bad --weights entry: {w}. Use format like 0.7,0.3")
        if pair not in unique_weights:
            unique_weights.append(pair)

    for w_sim, w_leaf in unique_weights:
        suffix = f"_w{int(round(w_sim*100)):02d}-{int(round(w_leaf*100)):02d}"

        # features for filtered vs unfiltered (scalers fit separately)
        feats_f = [c for c in NUMERIC_KEEP if c in pool.columns]
        feats_u = [c for c in NUMERIC_KEEP if c in cand.columns]

        compute_and_write(
            df_scope_filtered=pool,
            df_scope_unfiltered=cand,  # same scope, no date cutoff
            subj=subj,
            feats_filtered=feats_f,
            feats_unfiltered=feats_u,
            subject_pin=subject_pin,
            outdir=args.outdir,
            k=args.k,
            weights=(w_sim, w_leaf),
            suffix=suffix,
            ccao_df=ccao_df,
        )


if __name__ == "__main__":
    main()
