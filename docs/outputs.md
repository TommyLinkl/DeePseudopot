# Output Data Description

DeepPseudopot generates a rich set of artefacts as it initializes, trains, and exports pseudopotentials. Use this guide to understand what appears in the results directory and how to interpret each file.

## Initialization Artefacts
| File | Description |
| --- | --- |
| `oldFunc_plotBS.pdf` | Baseline band structure produced by the analytic Zunger potentials prior to neural-network fitting. |
| `initZunger_plotBS.pdf`, `initZunger_plotPP.pdf` | Diagnostics comparing initialized neural potentials against the reference data. |
| `initZunger_pot.dat`, `initZunger_qSpace_pot.dat` | Real- and reciprocal-space potentials derived from the initialized network. |
| `initZunger_PPmodel.pth`, `initZunger_epoch_<N>_PPmodel.pth` | Neural-network checkpoints saved during the optional Zunger pre-fit stage. |

## Training and Refinement Outputs
| File | Description |
| --- | --- |
| `epoch_<N>_plotBS.{pdf,png}`, `epoch_<N>_plotPP.{pdf,png}` | Snapshots of band-structure and potential convergence across epochs or Monte Carlo iterations. |
| `training_cost.dat`, `validation_cost.dat` | Numerical traces of objective values; suitable for plotting convergence trends. |
| `mc_iter_<N>_*` | Temporary records for each Monte Carlo proposal; retained only when monitoring in-flight runs. |
| `mc_checkpoint.pth`, `best_pot.*`, `best_plotPP.*` | Best-found models and potentials when Monte Carlo refinement is active. |

## Final Deliverables
| File | Description |
| --- | --- |
| `final_plotBS.pdf`, `final_plotPP.pdf` | Final visualizations for band structures and pseudopotentials. |
| `final_pot.dat`, `final_qSpace_pot.dat` | Exported potentials ready for downstream simulation pipelines. |
| `movie_BS.mp4`, `movie_PP.mp4` | Optional animations summarizing training progress; require `ffmpeg`. |

## Suggested Post-processing Workflow
1. Inspect `final_plotBS.pdf` for agreement with experimental or ab initio references.
2. Compare `final_pot.dat` against prior versions to quantify improvements.
3. Archive `training_cost.dat` and `validation_cost.dat` for regression tracking across tuning runs.
4. For Monte Carlo campaigns, promote `mc_checkpoint.pth` or `best_pot.*` into the next initialization bundle.

For real-time monitoring, restart strategies, and developer utilities, see [Monitoring & Utilities](outputs-monitoring.md). Troubleshooting tips for missing or malformed outputs live in the [Troubleshooting Guide](troubleshooting.md).
