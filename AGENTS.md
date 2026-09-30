# CP1: planar linkage synthesis

These instructions apply to this directory and its descendants. Work from this
directory when running CP1 commands; data paths in the notebooks are relative.

## Objective and constraints

Design planar linkages for all three curves in `kangaroo_target_curves.npy`.
Minimize both curve distance and material use, and maximize the final collection's
normalized Pareto hypervolume. A single mechanism with low curve error is not the
whole objective.

Use `LINKS/CP/__init__.py` as the source of truth for the current local evaluator.
Whenever the user asks to evaluate or score CP1 results, call the existing
`LINKS.CP.evaluate_submission` on the requested submission file or dictionary with
the supplied target curves. Pass the submission as provided and let that function
apply its evaluation rules. Report its returned scores as the authoritative result.
Do not replace or reimplement the evaluator, modify its scoring rules, or substitute
custom metrics or a separate hypervolume calculation unless explicitly requested.
Additional diagnostics must be labeled separately from the submission score. If
the existing evaluator cannot run, report the failure rather than substitute a score.

Import `REFERENCE_POINTS`, `SCORE_NORMALIZERS`, and `MAX_JOINTS` rather than copying
constants into optimization code. Current values are:

| Submission key | Target index and shape | Reference point: distance, material | HV normalizer |
| --- | --- | --- | --- |
| `Problem 1` | 0: round kangaroo | `(0.75, 10.0)` | 2.0 |
| `Problem 2` | 1: no ears or tail | `(1.2, 10.0)` | 1.5 |
| `Problem 3` | 2: full kangaroo | `(1.75, 20.0)` | 10.0 |

- At most 20 joints per mechanism and 1,000 submitted designs per target. The
  1,000-design limit is not an optimization-evaluation budget.
- The evaluator requires both objectives to be strictly below that target's
  reference point for a design to contribute. Missing or empty problems score zero.
- Overall score is the mean of the three normalized hypervolumes. Normalized
  scores can exceed 1; the normalizers are not upper bounds.
- Final evaluation uses `Tools(timesteps=200, max_size=20, material=True,
  scaled=False, device='cpu')`. Preserve this evaluation behavior. Diagnose any
  suspected evaluator bug separately from claims of optimizer improvement.
- Use the supplied distance and material calculations. Translation and rotation
  alignment do not imply scale invariance: physical rescaling changes both the
  curve comparison and material use. A paper's alternative metric is not the score.

## Reuse the supplied tools

- `2_155_Fall_26_CP1_Starter_Notebook.ipynb`: problem definition, fixed-topology
  NSGA-II example, gradient refinement, and submission examples.
- `Fall_26_CP1_Advanced_Starter_Notebook.ipynb`: mixed-variable topology search,
  random initialization, and refinement of a population.
- `LINKS.Optimization.Tools`: objective values and final candidate validation.
- `LINKS.Optimization.DifferentiableTools`: distance and material values plus both
  gradients with respect to initial joint coordinates `x0`. These gradients do not
  optimize discrete edges, joint types, or the output-joint choice.
- `LINKS.Optimization.MechanismRandomizer`: existing mechanism generation,
  including batch generation; inspect its interface before writing another sampler.
- `LINKS.Kinematics`, `LINKS.Geometry`, and `LINKS.Visualization`: simulation,
  curve comparison, mechanism plots, and Pareto plots.
- `LINKS.CP.make_empty_submission` and `evaluate_submission`: submission helpers.

Reuse instances, compilation, and supported batching. Account for JAX compilation
and warmup when comparing runtimes. Do not reimplement linkage derivatives merely
because a paper derives them. Check actual function signatures and return values;
some annotations and docstrings are less precise than the implementation.

## Search strategy and research context

- Follow the user's preference for non-ML optimization. Do not introduce trained
  models or external training-data requirements. Candidate clustering, if used,
  can operate on the population generated during the current optimization.
- Start from the existing baselines. Test improvements in topology representation,
  initialization, candidate diversity, local refinement, or hypervolume selection
  individually before combining them. Research ideas are hypotheses to test.
- Preserve useful distance-material tradeoffs. Distance-only refinement can improve
  the curve while worsening material or the collection's score. Use both objectives
  when deciding which candidates to refine and retain.
- Sancibrian et al. (2019), available in `papers/`, motivates elite-versus-cluster
  selection for local search. It studies fixed topologies and a single synthesis
  error; it does not supply CP1's topology search or multiobjective policy. Its
  hybrids improved accuracy in the examples but plain DE ran faster.
- SMS-EMOA motivates selection by hypervolume contribution. Gradient-derivation
  papers are background here because CP1 already supplies gradients. Consult
  `papers/README.md` for the inventory and read only papers relevant to the task.
- If clustering designs, compare compatible, consistently indexed representations
  and scale their variables appropriately. Do not compare arbitrary flattened
  coordinates from different topologies as though they describe the same variables.
- Near locking, gradients can become unreliable or non-finite. Check objective
  validity as well as gradients, retain the last valid design, and reduce or reject
  a bad step. Recheck candidates with `Tools`: a gradient failure alone does not
  establish that the mechanism cannot be simulated.

## Working and checking changes

- Keep changes small and readable. Reuse NumPy, SciPy, JAX, and pymoo already used
  here; avoid new frameworks and generic optimizer infrastructure without a need.
- Preserve existing notebook edits, outputs, downloaded papers, and saved candidate
  collections. Save experiments to distinct descriptive paths instead of silently
  overwriting `my_submission.npy` or `my_full_submission.npy`.
- Reuse an existing environment. For a new local environment, follow `readme.md`:
  Python 3.10, `uv venv --python 3.10 .venv`, then
  `uv pip install --python .venv/bin/python -r requirements.txt`.
  Respect the dependency pins. Do not rerun notebook clone/move/bootstrap cells
  against this existing checkout.
- Begin algorithm changes with a small, bounded run. Record the random seed,
  target, parameters, evaluation count, runtime, and per-target score. Compare
  methods under comparable budgets and distinguish a single run from repeated-run
  evidence. Separate compilation time from search time.
- Run one small, meaningful check for new optimization or submission logic. For a
  claimed improvement, evaluate the saved candidates with `evaluate_submission`
  and report its actual score breakdown. Documentation-only changes do not require
  rerunning optimization. State when dependencies or runtime prevent validation.
- Inspect representative generated curves and mechanisms as well as numerical
  scores. Explain what changed, its physical or optimization purpose, and the
  evidence supporting any improvement; do not invent benchmark results.

## Submission contract

Save one NumPy dictionary with keys `Problem 1`, `Problem 2`, and `Problem 3`, each
mapping to a list of mechanism dictionaries. Each mechanism contains:

| Field | Content |
| --- | --- |
| `x0` | Finite joint coordinates, shape `(N, 2)` |
| `edges` | Integer joint-index pairs, shape `(E, 2)` |
| `fixed_joints` | Integer fixed-joint indices, shape `(F,)` |
| `motor` | Two integer joint indices, shape `(2,)` |
| `target_joint` | Explicit integer output-joint index when one was selected |

Check index ranges, shapes, joint count, and simulation validity across the input
cycle. Preserve the chosen output joint through reordering and serialization;
omitting it or setting it to `None` invokes the solver's default selection.
Keep at most 1,000 useful designs per target; entries after the first 1,000 are
ignored. A complete submission should cover all three targets.

Use `np.save(output_path, submission)` and score the actual saved file through
`evaluate_submission(str(output_path), target_curves='kangaroo_target_curves.npy')`.
Report `Overall Score`, `Score Breakdown`, and `Normalized Score Breakdown`.
