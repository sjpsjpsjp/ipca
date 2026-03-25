# IPCA in Python

Python implementation of Instrumented Principal Components Analysis (IPCA),
building off of the estimator introduced in Kelly, Pruitt, and Su (2019).

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/sjpsjpsjp/ipca.git
```

Or clone and install in editable mode (for development):

```bash
git clone https://github.com/sjpsjpsjp/ipca.git
pip install -e ipca/
```

## Usage

```python
import ipca

# RZ is a DataFrame with a (Date, AssetID) MultiIndex.
# The first column is returns; remaining columns are characteristics.
model = ipca.ipca(RZ=RZ)

results = model.fit(K=3)           # fit with 3 latent factors

Gamma  = results['Gamma']          # characteristic loadings  (L x K)
Factor = results['Factor']         # latent factors           (K x T)
R2     = results['xfits']['R2_Total']   # total in-sample R²

# Out-of-sample estimation
results_oos = model.fit(K=3, OOS=True)
R2_oos = results_oos['rfits']['R2_Pred']
```

See the docstrings in `ipca.py` for full documentation of inputs and outputs.

## Citation

Kelly, Bryan T., Seth Pruitt, and Yinan Su (2019).
"Characteristics Are Covariances: A Unified Model of Risk and Return."
*Journal of Financial Economics* 134(3): 501–524.

## License

Copyright Seth Pruitt (2020–2026). All rights reserved.
