# Output Data Description

DeepPseudopot generates a rich set of artefacts as it initializes, trains, and exports neural network pseudopotentials. Use this guide to understand what appears in the results directory and how to interpret each file.

## Initialization Artefacts

| File | Description |
| --- | --- |
| `oldFunc_plotBS.pdf` | Baseline band structure produced by the analytic Zunger potentials prior to neural-network fitting. |
| `initZunger_plotBS.pdf`, <br>`initZunger_plotPP.pdf` | Diagnostics comparing initialized neural potentials against the reference data in its prediction of band structure and pseudopotential itself. |
| `initZunger_pot.dat`, <br>`initZunger_qSpace_pot.dat` | Real- and reciprocal-space potential files derived from the initialized network. |
| `initZunger_PPmodel.pth`, <br>`initZunger_epoch_<N>_PPmodel.pth` | Neural-network checkpoints saved during the optional Zunger pre-fit stage. |

## Training and Refinement Outputs

| File | Description |
| --- | --- |
| `epoch_<N>_plotBS.{pdf,png}`, <br>`epoch_<N>_plotPP.{pdf,png}` | Snapshots of band-structure and potential convergence during training epochs or Monte Carlo iterations. |
| `training_cost.dat`, <br>`validation_cost.dat` | Data files of the loss values; suitable for plotting and inspecting convergence trends. |
| `mc_iter_<N>_*` | Temporary records for each Monte Carlo proposal; retained only when monitoring in-flight runs. |
| `mc_checkpoint.pth`, <br>`best_pot.*`, <br>`best_plotPP.*` | Best-found models and potentials when Monte Carlo refinement is active. |

## Final Deliverables

| File | Description |
| --- | --- |
| `final_plotBS.pdf`, <br>`final_plotPP.pdf` | Final visualizations for last-epoch (or best) band structures and pseudopotentials. |
| `final_pot.dat`, <br>`final_qSpace_pot.dat` | Exported potentials ready for downstream simulation pipelines. |
| `movie_BS.mp4`, <br>`movie_PP.mp4` | Optional animations summarizing training progress; fun for visualization. |

## Suggested Post-processing Workflow

1. Inspect `final_plotBS.pdf` for agreement with experimental or ab initio references.
2. Compare `final_pot.dat` against prior versions to quantify improvements.
3. Archive `training_cost.dat` and `validation_cost.dat` for regression tracking across tuning runs.
4. For Monte Carlo campaigns, promote `mc_checkpoint.pth` or `best_pot.*` into the next initialization bundle.

For real-time monitoring, restart strategies, and troubleshooting tips, see [Troubleshooting Guide](troubleshooting.md).
