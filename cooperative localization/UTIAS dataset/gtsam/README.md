# UTIAS localization with GTSAM

Run with Python 3.14, where GTSAM 4.3.0 is installed:

```bash
/usr/local/bin/python3 "/home/cora-tuf/GaBP_Tutorial/cooperative localization/UTIAS dataset/gtsam/UTIAS dataset.py"
```

Add `--no-plot` to run without a window.

- `UTIAS dataset.py`: experiment settings, LM optimization, history, RMSE and plot.
- `build_gtsam_graph.py`: initial poses, priors, odometry and range-bearing factors.
- `history.csv`: initial poses and each LM iteration; generated when running.

Edit `hz`, `start_offset`, `duration`, `max_iters`, `options` and the covariance
matrices at the top of `UTIAS dataset.py`. Defaults match the parent GBP script.
`options=1` uses all observations, `2` uses landmarks only, and `3` uses robots only.
Priors and odometry are included in every mode.

The parent data loader, odometry integration, first-pose interpolation, observation
residuals/Jacobians and plotting code are shared. Only robot poses are variables;
landmark coordinates are fixed. Every measurement uses the same nearest time node.
Range-bearing covariance order is `[range, bearing]`; odometry covariance is the
final right-tangent pose-error covariance, without extra time scaling.

Use **GTSAM >= 4.3**: its native Pose2 prior/between factors use the same SE(2)
log-map residuals as this GBP problem. The optimizer uses LM error tolerances
(`1e-6` absolute/relative), so its iteration count and stopping rule differ from GBP.
The printed error is `0.5 * sum(r.T @ inv(cov) @ r)`, including priors.

History contains `iteration,variable_id,mean`, with a 3x3 pose matrix in `mean`.
It can be read by the parent plotter's `load_history`; covariance is not recorded.
Robot/variable ordering matches GBP. This history is written inside this folder.
