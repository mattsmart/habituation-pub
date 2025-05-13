# Minimal motifs for habituating systems

Code for the paper ["Minimal motifs for habituating systems" (PNAS 2024)](https://www.pnas.org/doi/10.1073/pnas.2409330121).

Habituation refers to the ubiquitous ability of organisms -- from animals to single cells -- to tune out repetitive stimuli. This repository implements and analyzes minimal dynamical systems exhibiting various hallmarks of habituation, including:

- Short-term response decrements
- Spontaneous recovery
- Dishabituation
- Frequency-dependent effects
- Long-term memory formation

## Dependencies
- Python >=3.9 with libraries listed in `requirements.txt`  
- Install dependencies: `pip install -r requirements.txt`

## Repository Structure

- `src/`: Core implementation of habituation models and utilities
- `src/notebooks/`: Jupyter notebooks for reproducing figures and analysis

Recreating panels of figures appearing in the text:  
- Figure 2 - `notebooks/figures_adapt_vs_hab.ipynb`
- Figure 3D - `notebooks/figures_minimal_unit.ipynb`
- Figure 3E - `notebooks/figures_minimal_unit.ipynb`
- Figure 3F - `notebooks/figures_hallmarks.ipynb`
- Figure 4 - `notebooks/figures_hallmarks.ipynb`
- Figure 5 - `notebooks/figures_hallmarks.ipynb`
- Figure 6 - `notebooks/figures_hallmarks.ipynb`
- Figure S1 - `notebooks/figures_hallmarks.ipynb`  

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{smart2024pnas,
    author = {Matthew Smart  and Stanislav Y. Shvartsman  and Martin Mönnigmann},
    title = {Minimal motifs for habituating systems},
    journal = {Proceedings of the National Academy of Sciences},
    volume = {121},
    number = {41},
    pages = {e2409330121},
    year = {2024},
    doi = {10.1073/pnas.2409330121},
    URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2409330121},
    eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2409330121},
}
```
