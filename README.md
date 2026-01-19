# Active Pre-Training for Generative Discovery

This project makes extensive use of [flowgym](https://github.com/cristianpjensen/flowgym). 

Example usage:
```bash
pixi run python -m active_pretraining.run toy --dir experiments/toy

pixi run python -m active_pretraining.run mnist --dir experiments/mnist

pixi run python -m active_pretraining.run qm9 --dir experiments/qm9

pixi run python -m active_pretraining.run geom_drugs --dir experiments/geom_drugs

pixi run python -m active_pretraining.run stable_diffusion --dir experiments/stable_diffusion
```

Once a training run is done, you can compute sample metrics separately using:
```bash
pixi run python -m active_pretraining.compute_sample_metrics <dir>
```
