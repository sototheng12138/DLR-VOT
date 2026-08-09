# Data

## Public release boundary

The transaction ledger and the completed daily railway freight table are not included in this repository. They contain operational observations supplied for research, and public release requires authorization from the data owner. Pointwise labels, predictions, event scores, and diagnostic arrays are excluded for the same reason.

The experiments on the paper data can therefore be rerun only by an authorized user. This repository still provides the model input contract, a synthetic generator, integrity checks, the chronological split, and aggregate result tables.

## Model input

The forecasting code starts from a completed daily table. It does not convert the transaction ledger into that table. The expected CSV has:

- a first column named `date` or `日期`;
- four subsequent numeric commodity columns;
- one row per calendar day, in chronological order;
- no duplicate or missing dates;
- no missing or negative values.

Dates may use `YYYY-MM-DD` or the `YYYY/M/D` form found in the executed paper table. Values remain in the units recorded in the source data because a physical unit was not supplied with the extract.

After daily aggregation and calendar completion, zero is assigned when no matching record exists for a commodity and date in the supplied extract. It does not establish that no physical movement occurred outside the extract.

Some historical code identifiers use `has_shipment`. They are retained for checkpoint compatibility and refer only to a positive recorded aggregate, not to independently verified physical movement.

The smallest positive value in the completed table is 66,640 in the source units. The historical event code uses `y > 1`, which therefore gives exactly the same labels as `y > 0` for this table.

The completed table used in the study has 1,096 daily rows from 2023-01-01 through 2025-12-31. Its SHA-256 is:

```text
2796e15d49b2dac0ad0dd58d6800d347a43bdf7336213059005e474b47093860
```

The checksum identifies an authorized local copy; it is not a download link.

## Chronological split

For a table with 1,096 rows, the loader assigns 767 days to training, 110 to validation, and 219 to testing. With an input length of 96 and forecast horizon of 48, this gives 624 training origins, 63 validation origins, and 172 test origins. The validation and test loaders retain the preceding 96 days as context. Standardization is fitted on the first 767 rows only.

The loader follows CSV row order and does not sort dates. Run the validator before training.

## Synthetic example

The synthetic generator creates a valid table without copying the railway observations:

```bash
python scripts/generate_synthetic_data.py \
  --daily-output /tmp/dlr_vot_daily.csv \
  --event-output /tmp/dlr_vot_validation.csv
python scripts/validate_daily_data.py /tmp/dlr_vot_daily.csv
```

Synthetic values are intended for tests and interface checks. They do not reproduce any empirical result in the paper.

## Using an authorized copy

Place the completed table at `dataset/2023_2025_Iron_data.csv` or pass another path to the scripts. Do not commit it. Before a run, record its checksum:

```bash
sha256sum dataset/2023_2025_Iron_data.csv
```

This release does not include a program that converts the ledger into the daily table. The source ledger, provider rules, and full preprocessing program were not retained in a form that supports exact reconstruction.
