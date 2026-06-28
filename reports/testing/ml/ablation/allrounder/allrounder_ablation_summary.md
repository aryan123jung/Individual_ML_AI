# Allrounder Ablation Summary

This test measures whether the allrounder model remains strong after removing composite features that may be closely aligned with the label.

- full_feature_set: accuracy `0.9427`, F1 `0.9565`, ROC-AUC `0.9905`.
- without_composite_strength_scores: accuracy `0.9427`, F1 `0.9568`, ROC-AUC `0.9908`.
- without_allrounder_index: accuracy `0.8889`, F1 `0.9136`, ROC-AUC `0.9602`.
- raw_features_only: accuracy `0.8530`, F1 `0.8889`, ROC-AUC `0.9365`.