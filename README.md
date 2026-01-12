# Active Pre-Training for Generative Discovery

This project makes extensive use of [flowgym](https://github.com/cristianpjensen/flowgym). 

Example usage:
```bash
pixi run python -m active_pretraining.run --problem_setup toy --dir experiments/toy

pixi run python -m active_pretraining.run --problem_setup mnist --dir experiments/mnist

pixi run python -m active_pretraining.run --problem_setup qm9 --dir experiments/qm9
```

Once a training run is done, you can compute sample metrics separately using:
```bash
pixi run python -m active_pretraining.compute_sample_metrics <dir>
```
