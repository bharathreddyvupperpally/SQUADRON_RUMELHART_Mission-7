# AURA squad submission 7 — Mission 7: Sagittarius A* (SEC-07-SGR)

**Test-Time Compute search stack:** Tree-of-Thoughts, Monte Carlo Tree Search with PUCT, a Process Reward Model with
Math-Shepherd labels, and an Event Horizon Pruner that defends against PRM reward hacking (Ship Anomaly 07).

Course: Conversational AI — Winter Semester 2026/27 · Squad RUMELHART · USSE Noosphere Bridge Crew

Pure Python (standard library only, 3.9+). No installs, no network, no API keys. Every run is deterministic (fixed seeds).

## Repository layout

| Folder | Station | Main files |
|---|---|---|
| `01_Data_Specialist` | Process Reward Model and Math-Shepherd | `prm_evaluator.py`, `evaluation_demo.py`, `test_prm_evaluator.py`, `sample_trajectories.jsonl` |
| `02_Protocol_Specialist` | 4-phase MCTS with PUCT | `mcts_protocol.py`, `nav_domain.py`, `test_mcts_protocol.py` |
| `03_Security_Sentinel` | Event Horizon Pruner, reward-hacking defence | `horizon_pruner.py`, `anomaly_07_demo.py`, `test_horizon_pruner.py` |
| `04_Systems_Architect` | Tree-of-Thoughts and integration | `tot_solver.py`, `nav_domain.py`, `run_mission.py`, `test_tot_solver.py`, `test_integration.py`, `architecture_diagram.dot` |
| repository root | Full squad | `ps07_solution.ipynb`, `README.md`, `requirements.txt`, `COMMANDS FOR MISSION-7.txt`, `AURA_File_Guide_M7.xlsx`, final report (`.docx`, `.pdf`) |

The four station folders must stay side by side, because `run_mission.py`, `test_integration.py` and the notebook import
from all of them. Each station also runs on its own.

## Run

```bash
(cd 01_Data_Specialist     && python evaluation_demo.py)
(cd 02_Protocol_Specialist && python mcts_protocol.py)
(cd 03_Security_Sentinel   && python anomaly_07_demo.py)
(cd 04_Systems_Architect   && python tot_solver.py)
(cd 04_Systems_Architect   && python run_mission.py)      # integrated experiment
```

Unit tests (45): run `python test_<module>.py` inside each folder, or open `ps07_solution.ipynb` in Jupyter from the
repository root (`pip install -r requirements.txt`). All commands, including the Git push steps, are in
`COMMANDS FOR MISSION-7.txt`.

## What each station does

* **Systems Architect:** `ThoughtNode` and `TreeOfThoughtsSolver` with `bfs_beam_search()` (beam width B per depth) and
  `dfs_backtrack_search()` under a hard `max_expansions` budget; a `prune_fn` hook lets the pruner sever branches.
* **Protocol Specialist:** `MCTSNode` and `MCTSEngine`: selection by exact `PUCT = Q + c_puct·P·√N(s)/(1+N(s,a))`,
  expansion with LLM-style priors, seeded rollouts, backpropagation; clean zero-visit handling; PUCT traces across `c_puct`.
* **Data Specialist:** `ProcessRewardEvaluator` with step scores and `min` / `prod` / `geometric_mean` reductions, an
  Outcome Reward Model for comparison, a false-positive trap benchmark, and Math-Shepherd soft/hard labels from M rollouts.
* **Security Sentinel:** `EventHorizonPruner` severs a branch as soon as `r_PRM < tau_horizon`, detects sycophantic
  filler and step-splitting inflation, and logs pruned branches and tokens saved.

## Status and assumptions

* All code and tests were executed; every number in the report is a real output of the delivered scripts.
* The environment is our own toy navigation domain (arithmetic thoughts, start 2, target 61), and the proposer can hallucinate.
  The PRM is a transparent rule-based stand-in for a learned model, and the reward-hacking attacks are simulated.
  Results show how the components behave; they do not predict performance with a real language model.
* The instructor file `ps07.md` was not opened directly; the work follows the mission brief text.
* No language model is called.

Report: `SQUADRON_RUMELHART-MISSION-7_FINAL_REPORT.pdf`.
