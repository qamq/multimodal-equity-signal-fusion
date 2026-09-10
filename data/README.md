# Data

No licensed datasets, private expert predictions or trained models are distributed.

This directory is ignored except for this README and two opt-in locations:

- `synthetic/`: small, intentionally public generated examples.
- `fixtures/`: small, intentionally public validation fixtures.

Neither directory currently contains data. The main demo generates its synthetic panel in memory. Keep CRSP, Compustat, WRDS, Bloomberg and other restricted inputs, prediction outputs and checkpoints outside these public locations.

See [Data contracts](../docs/DATA.md) for required columns and identifier/target-timing rules.
