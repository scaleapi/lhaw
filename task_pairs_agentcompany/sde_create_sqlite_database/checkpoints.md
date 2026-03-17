# Checkpoints

**LHAW version**: 8 checkpoints, **6 pts max** (CP2 & CP3 = 0pts).

## Checkpoint 1 (1pt)

Navigated to the file on owncloud.

## Checkpoint 2 (0pts) ⚠️

~~Downloaded the `July-September 2024 Financials.ods` file.~~

**N/A in LHAW**: Files are pre-staged in `/inputs/THE OWNCLOUD FOLDER/`. Worth 0 points.

## Checkpoint 3 (0pts) ⚠️

~~The sqlite3 package is installed.~~

**N/A in LHAW**: Python's `sqlite3` is a standard library module and is always available. Worth 0 points.

## Checkpoint 4 (1pt)

There is a local sqlite database with the name "financial_report.db".

## Checkpoint 5 (1pt)

The sqlite database has a table called "financial_categories" with the following columns:

- category_id
- category_name
- category_type

## Checkpoint 6 (1pt)

The sqlite database has a table called "financial_details" with the following columns:

- detail_id
- category_id
- month
- actual
- budget
- variance
- percent_of_budget

## Checkpoint 7 (1pts)

A SQL query to get the minimum variance of "Total Income" returns "2024-08" for the month, and -35000 for the variance.

## Checkpoint 8 (1pts)

A SQL query to get the sum of "actual" for the categories "Software Development Services" and "Consulting" for "2024-07" returns 95000.
