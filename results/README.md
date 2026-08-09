# Reported tables

`tables` contains the aggregate values reported in Tables 1–19 of the manuscript. These files do not contain dates, pointwise observations, predictions, or event scores.

Run the following command to check the file set, checksums, and principal locked values:

```bash
python scripts/verify_reference_tables.py
```

The tables are reported outputs. Without the restricted observations and pointwise model outputs, this public repository checks their integrity but does not regenerate them from raw data.
