# Building Retail Features

How raw transactions become leakage-safe, point-in-time customer snapshots — the two-stage process behind `data/processed/retail_training_snapshots.csv`.

Read `PLATFORM_ARCHITECTURE.md` first for how this fits into the overall data flow.

---

## Stage 1: Get the raw data

```bash
python3 pipelines/download_retail_data.py
```

Downloads the UCI "Online Retail II" dataset (Chen, D. 2012, CC BY 4.0,
[archive.ics.uci.edu/dataset/502](https://archive.ics.uci.edu/dataset/502/online+retail+ii)):
1,067,371 line-item transactions from a UK-based online retailer, Dec 2009
– Dec 2011. Extracts the source `.xlsx` (two sheets, one per year) and
concatenates them into `data/raw/online_retail_II.csv` for fast reloads.
Idempotent — if the CSV already exists, it does nothing. The raw file is
gitignored; this script is how anyone reproduces it.

Raw columns: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`.

## Stage 2: Build point-in-time snapshots

```bash
python3 pipelines/build_retail_features.py
```

### Cleaning (`load_transactions`)

- Rows with no `Customer ID` can't be attributed to a customer → dropped.
- Non-positive `Price` (postage adjustments, bad-debt write-offs, free
  items) is a data error → dropped.
- Invoices starting with `C` are cancellations → **kept**, flagged
  `is_return=True`. Returns carry real signal (they predict churn), so
  they stay in the feature calculations rather than being dropped.

Kept 824,293 of 1,067,371 raw rows (5,939 customers, Dec 2009 – Dec 2011)
on the last full run.

### Snapshotting (`build_snapshot`)

One row per `(customer_id, snapshot_date)`. The core discipline:

```
features  <- computed ONLY from transactions strictly BEFORE snapshot_date
labels    <- computed ONLY from transactions AFTER (and including)
             snapshot_date, each over its own forward window
```

Only customers with at least one transaction before the snapshot are
eligible — you can't score a customer who doesn't exist yet. A customer
whose first purchase is after the snapshot is silently excluded from that
snapshot's rows (see `test_customers_without_history_are_excluded`).

**The 7 features** (`RETAIL_FEATURES` in code), all numeric:

| Feature | Window | Meaning |
|---|---|---|
| `recency_days` | lifetime | Days since the customer's most recent purchase, as of the snapshot |
| `tenure_days` | lifetime | Days since the customer's first-ever purchase, as of the snapshot |
| `frequency_90d` | trailing 90 days | Distinct invoices (non-return) in the window |
| `monetary_90d` | trailing 90 days | Total line revenue (non-return) in the window |
| `avg_basket_value_90d` | trailing 90 days | `monetary_90d / frequency_90d` (0 if no purchases in the window) |
| `distinct_products_90d` | trailing 90 days | Distinct `stock_code`s purchased in the window |
| `return_rate_90d` | trailing 90 days | Returned line items ÷ all line items in the window (0 if no activity) |

**Two labels** (`TARGET_COLUMNS` in code), both binary, both derived from
the same future transactions but over different horizons — see
[[split_once_reuse_architecture]] for why they can't be used as features
for each other:

| Label | Horizon | Meaning |
|---|---|---|
| `churn_label_180d` | 180 days after snapshot | 1 = **no** non-return purchase in the window (churned) |
| `propensity_label_30d` | 30 days after snapshot | 1 = **a** non-return purchase in the window (near-term propensity) |

They deliberately measure different things: a customer can be "not
churned" over 180 days while still having zero 30-day propensity (e.g.
they repurchase on day 45, just outside the short window) —
`test_propensity_reflects_short_window` in
`tests/test_retail_features.py` asserts exactly this.

### Multi-snapshot assembly (`build_multiple_snapshots` / `main`)

One snapshot per calendar month (`freq="MS"`), `--start-date 2010-03-01` to
`--end-date 2011-06-01` by default. A snapshot is skipped automatically if
the data doesn't extend far enough forward for a complete 180-day churn
label (protects against a truncated, silently-wrong label near the end of
the observable data).

Last full run: **16 monthly snapshots, 59,162 rows, 4,996 unique
customers** — 44.94% churn rate, 21.94% propensity rate overall (both vary
by month; see the per-snapshot printout when you run the script).

Output: `data/processed/retail_training_snapshots.csv`.

### Validation

`validate_training_dataset()` checks, before writing anything:
- No duplicate `(customer_id, snapshot_date)` rows.
- No missing values in any feature or label column.
- Both labels are strictly 0/1.

---

## Next step

`pipelines/prepare_retail_ml_dataset.py` turns this snapshot table into
chronological train/validation/test splits per target — see
`ML_DATASET_REFERENCE.md`.
