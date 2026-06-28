import numpy as np
import pandas as pd


def numeric_matrix(df, columns, fill_values=None):
    data = df[columns].copy()
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        fill_value = fill_values.get(column, 0) if fill_values else 0
        data[column] = data[column].fillna(fill_value)
    return data.astype(float)


def build_scaler(reference_df, columns):
    ref = numeric_matrix(reference_df, columns)
    means = ref.mean()
    stds = ref.std().replace(0, 1).fillna(1)
    return means, stds


def scaled_matrix(df, columns, means, stds):
    values = numeric_matrix(df, columns)
    return ((values - means) / stds).fillna(0).to_numpy()


def cosine_similarity_matrix(left_matrix, right_matrix):
    left_norm = np.linalg.norm(left_matrix, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right_matrix, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1
    right_norm[right_norm == 0] = 1
    return (left_matrix / left_norm) @ (right_matrix / right_norm).T


def best_role_matches(smat_df, ipl_df, role_column, feature_columns, comparison_type, top_n=3):
    smat_df = smat_df.copy()
    ipl_df = ipl_df.copy()

    if "eligible_for_model" in smat_df.columns:
        smat_df = smat_df[smat_df["eligible_for_model"]].copy()
    if "eligible_for_model" in ipl_df.columns:
        ipl_df = ipl_df[ipl_df["eligible_for_model"]].copy()

    rows = []
    role_overlap = set(smat_df[role_column].dropna()) & set(ipl_df[role_column].dropna())
    for role in sorted(role_overlap):
        smat_role = smat_df[smat_df[role_column] == role].reset_index(drop=True)
        ipl_role = ipl_df[ipl_df[role_column] == role].reset_index(drop=True)
        if smat_role.empty or ipl_role.empty:
            continue

        reference = pd.concat([smat_role[feature_columns], ipl_role[feature_columns]], ignore_index=True)
        means, stds = build_scaler(reference, feature_columns)
        sims = cosine_similarity_matrix(
            scaled_matrix(smat_role, feature_columns, means, stds),
            scaled_matrix(ipl_role, feature_columns, means, stds),
        )

        for smat_idx, smat_row in smat_role.iterrows():
            top_indices = np.argsort(sims[smat_idx])[::-1][:top_n]
            for rank, ipl_idx in enumerate(top_indices, start=1):
                ipl_row = ipl_role.iloc[ipl_idx]
                rows.append(
                    {
                        "comparison_type": comparison_type,
                        "role": role,
                        "smat_player": smat_row["player"],
                        "smat_team": smat_row.get("team"),
                        "ipl_player": ipl_row["player"],
                        "ipl_team": ipl_row.get("team"),
                        "similarity_rank": rank,
                        "similarity_score": round(float(sims[smat_idx, ipl_idx]) * 100, 2),
                    }
                )

    return pd.DataFrame(rows)


def best_similarity_lookup(similarity_df):
    if similarity_df.empty:
        return {}
    best = similarity_df[similarity_df["similarity_rank"] == 1]
    return {
        (row["comparison_type"], row["role"], row["smat_player"]): row
        for _, row in best.iterrows()
    }
