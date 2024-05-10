Release notes
=============

0.2.1 (2024-05-06)
-------------
* Added changepoint_sensitivity instead of pen parameter. NOw, the pen parameter is estimated automatically
* Added support for ROCKET-based clustering with sktime 0.29.0
* Bugfix with incorrect visualization reported in #9
* Bugfix with internal data not updating after transform (this is needed for the visualization)
* Added feature of dynamic cluster determination when float is provided as n_clusters parameter


0.2.0 (2024-04-05)
-------------
* Change the way the cluster number is detected. Now it is detected based on the average number of breakpoints found in the dataset
* Added parameters that allow to specify SHAP aggregation function for weight calculation
* Added function that allows to incorporate cluster starts in window as features.
* Updated documentation
* Bugixes


0.1.3 (2024-03-05)
-------------
* Modularization of code
* Scikit-learn interface added
* Documentation updated
* Examples added

0.1.0 (2024-02-23)
-------------
* Added pypl installation
* Added sphinx documentation