# Optical SEM Local Nadaraya-Watson Smoother Comparison

Datasets found: **39**
Datasets processed: **39**

The local smoothing models are Nadaraya-Watson smoothers on the doubled-angle line embedding `(cos 2 phi, sin 2 phi)`. The three kernels are Gaussian RBF, uniform/local-neighbourhood, and multiplicative RBF-von-Mises. Hyperparameters are selected by spatially blocked held-out axial RSS.

## Kernel Summary

| model | kernel | datasets | mean_test_rss_mean | mean_test_rss_median | mean_test_mae_deg_mean | bandwidth_median | kappa_median |
|---|---|---|---|---|---|---|---|
| RBF-von-Mises NW smoother | multiplicative_rbf_vm | 39 | 0.1746 | 0.17 | 16.7209 | 10.5378 | 32 |
| RBF NW smoother | rbf | 39 | 0.2206 | 0.227 | 19.6959 | 5.7387 |  |
| Uniform NW smoother | uniform | 39 | 0.2312 | 0.2379 | 20.524 | 14.2796 |  |

## Selected NW Smoother Summary

| model | datasets | mean_test_rss_mean | mean_test_rss_median | mean_test_mae_deg_mean |
|---|---|---|---|---|
| RBF-von-Mises NW smoother | 39 | 0.1746 | 0.17 | 16.7209 |
| RBF NW smoother | 39 | 0.2206 | 0.227 | 19.6959 |
| Uniform NW smoother | 39 | 0.2312 | 0.2379 | 20.524 |

## Plots

![Mean RSS](plots/nw_smoother_mean_rss.png)

![NW test distribution](plots/nw_smoother_boxplot.png)

![NW winners](plots/nw_smoother_winner_counts.png)

![EE+0p2-data-225x225/EE_0.2data_rot_-22deg/arrow_segments.mat](plots/streamline_overlays/EE+0p2-data-225x225__EE_0.2data_rot_-22deg__arrow_segments_nw_streamlines.png)

![EE+0p22-data-225x225/EE_0.22data_rot_35.5deg/arrow_segments.mat](plots/streamline_overlays/EE+0p22-data-225x225__EE_0.22data_rot_35.5deg__arrow_segments_nw_streamlines.png)

![EE+0p23-data-225x225/EE_0.23data_rot_0deg/arrow_segments.mat](plots/streamline_overlays/EE+0p23-data-225x225__EE_0.23data_rot_0deg__arrow_segments_nw_streamlines.png)

![EE+0p4-data-225x225/EE_0.4data_rot_-67deg/arrow_segments.mat](plots/streamline_overlays/EE+0p4-data-225x225__EE_0.4data_rot_-67deg__arrow_segments_nw_streamlines.png)

![EE+0p42-data-225x225/EE_0.42data_rot_-7.5deg/arrow_segments.mat](plots/streamline_overlays/EE+0p42-data-225x225__EE_0.42data_rot_-7.5deg__arrow_segments_nw_streamlines.png)

![EE+0p43-data-225x225/EE_0.43data_rot_14.5deg/arrow_segments.mat](plots/streamline_overlays/EE+0p43-data-225x225__EE_0.43data_rot_14.5deg__arrow_segments_nw_streamlines.png)
