# AlphaRouter

AlphaRouter is a dynamic portfolio routing system that uses a Transformer model to learn causal daily utility labels and perform preference-conditioned online risk-and-tail-aware allocation across multiple markets.

## Project Structure

- Alpharouter/
  - model.py - Implements the Transformer router and shared per-expert evidence scorer.
  - experiment.py - Builds causal daily utility labels, trains independent routers, and selects checkpoints using validation data.
  - committee_experiment.py - Calibrates dynamic share and switching margin on validation data, and applies unanimous hard consensus.
  - inal_report.py - Produces comparison tables and evaluation metrics.
- 	ests/ - Comprehensive test suite for causal windows, daily turnover labels, simplex feasibility, and disagreement fallback.

## Installation

Install the required dependencies:

`bash
pip install -r requirements.txt
`

## Data Availability

The datasets are not included in this repository to keep it lightweight. The experiments run on four public proxy panels (DJIA29, HSI56, CSI98, CRYPTO18) constructed using the AlphaGAT methodology.

**How to get the data:**
1. Download the pre-processed .npz files (e.g., Alphagat_djia29_frozen_extended_20260808.npz and lphagat_djia29_public_2010_2023.npz) from the official AlphaGAT data releases or public drive links.
2. Create a folder named data/ in the root of this project.
3. Place all downloaded .npz files into the data/ directory.

The Alpharouter/config.py file maps each market configuration to the expected filenames in the data/ directory.

## Running the Experiments

To fully reproduce the AlphaRouter experiments, run the following commands sequentially:

1. **Train the models** (Trains 5 independent routers across 4 datasets):
   `bash
   python -m alpharouter.experiment --seeds 101 202 303 404 505 --epochs 100 --device auto
   `

2. **Run committee evaluation**:
   `ash
   python -m alpharouter.committee_experiment --seeds 101 202 303 404 505 --device auto
   `

3. **Generate final report**:
   `bash
   python -m alpharouter.final_report
   `

## Testing

Ensure code correctness by running the tests. The tests use deterministically generated synthetic data, so you can run them even without downloading the actual datasets:

`bash
pytest -q tests/test_alpharouter_dynamic.py
`

## Technical Details

The selected architecture has hidden dimension 128, six Transformer layers, eight attention heads, and feed-forward dimension 256. The final training budget is at most 100 epochs, with validation-selected checkpoints. 

AlphaRouter is designed to evaluate neural routing benefits across different market regimes (DJIA29, HSI56, CSI98, CRYPTO18) while maintaining strict adherence to causality, risk constraints, and test holdouts.
