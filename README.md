# DeepMIMO
DeepMIMO Toolchain: Bridging RayTracers and Simulators (Matlab &amp; Python)

**[Goal]** Enabling large-scale AI benchmarking using site-specific wireless raytracing datasets.

**[How]** Converting the outputs of the best ray tracers in wireless to a distributable format readable by the best simulation toolboxes. 

```
deepmimo/
├── converter/
│   ├── aodt/
│   ├── sionna_rt/
│   └── wireless_insite/
└── generator/
    ├── python/
    └── matlab/
```

# Getting Started

Install:
```bash
pip install deepmimo
```

Generate a dataset:
```python
import deepmimo as dm
dm.summary('asu_campus')  # Print summary of the dataset
dm.generate('asu_campus')  # Load and generate the dataset
```

Try it out on Colab: ......

# Building the docs

| Step    | Command                                           | Description                       |
|---------|---------------------------------------------------|-----------------------------------|
| Install | `pip install sphinx sphinx-rtd-theme`             | Install required dependencies     |
| Build   | `cd docs`<br>`sphinx-build -b html . _build/html` | Generate HTML documentation       |
| Serve   | `cd docs/_build/html`<br>`python -m http.server`  | View docs at http://localhost:8000|

# Contributing

To contribute, fork the repository and make a pull request. It will get a reply in <24 hours. 

Converter: Our priority is maximum feature coverage on the supported ray tracers. 

Generator: We are always looking for what tools are helpful to users. If you used DeepMIMO (e.g., in a paper) and had to create several custom tools to visualize, process, or analyze data, feel free to open a pull request! 

# Citation

If you use this software, please cite it as:

```bibtex
@misc{alkhateeb2019deepmimo,
      title={DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications}, 
      author={Ahmed Alkhateeb},
      year={2019},
      eprint={1902.06435},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/1902.06435}, 
}
```
