# ML Evaluation Summary

## Overview
The player recommendation system was evaluated using a supervised learning stage and an unsupervised clustering stage.
For supervised learning, historical IPL feature sets were split into training and testing partitions using an 80:20 train/test split with a fixed random state of 42.
Two classification algorithms were compared in each domain: Logistic Regression and Random Forest.

## Supervised Model Results
- Allrounder: the best model was `random_forest` with accuracy `0.9497`, F1-score `0.9540`, and ROC-AUC `0.9881`.
- Batting: the best model was `random_forest` with accuracy `0.8632`, F1-score `0.8785`, and ROC-AUC `0.9240`.
- Bowling: the best model was `random_forest` with accuracy `0.9000`, F1-score `0.8974`, and ROC-AUC `0.9600`.

These results indicate that the Random Forest model consistently produced the strongest predictive performance across batting, bowling, and all-rounder evaluation. This suggests that nonlinear relationships between engineered cricket features are important for predicting IPL suitability.

## Confusion Matrix Interpretation
- Batting: true negatives `35`, false positives `8`, false negatives `5`, and true positives `47`.
- Bowling: true negatives `37`, false positives `4`, false negatives `4`, and true positives `35`.
- Allrounder: true negatives `68`, false positives `5`, false negatives `3`, and true positives `83`.

The confusion matrices show that the models are not only accurate overall, but also effective at identifying successful players without producing excessive misclassification. This is important because the system is intended as a decision-support tool for selectors and coaches.

## Clustering Results
- Batting: `3` clusters were selected with silhouette score `0.2213`.
- Bowling: `3` clusters were selected with silhouette score `0.2058`.
- Allrounder: `6` clusters were selected with silhouette score `0.3224`.

The clustering stage supports the supervised model by grouping players into role-based archetypes. Among the three domains, all-rounder clustering achieved the strongest separation, indicating clearer natural structure in all-rounder profiles than in batting-only or bowling-only profiles.

## Thesis Interpretation
Overall, the evaluation demonstrates that the proposed system can learn meaningful patterns from historical IPL data and apply them to domestic player assessment. The combination of feature engineering, supervised classification, and unsupervised clustering strengthens the system's credibility as an AI/ML-based player recommendation framework for IPL squad formation.