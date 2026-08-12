"""
experiment_measurement_sql.py

Generates SQL for an experiment measurement workflow:

1. Minimum Detectable Effect (MDE)
2. Test/Control balance using Standardized Mean Difference (SMD)
3. ANCOVA-ready customer dataset:
      - pre-period behavior
      - treatment/control assignment
      - post-period observed outcome

The final ANCOVA regression is normally run in Python because SQL is best
used to construct the analysis dataset, while statsmodels is better suited
for statistical estimation.

Assumptions
-----------
- Customer-level randomized experiment
- One row per customer in experiment assignment table
- Outcome data can be aggregated by customer and time period
- treatment = 1 for Test, 0 for Control
"""

from dataclasses import dataclass
from textwrap import dedent


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class ExperimentConfig:
    assignment_table: str = "analytics.experiment_assignment"
    outcome_table: str = "analytics.customer_outcomes"
    customer_id: str = "customer_id"

    treatment_column: str = "treatment"
    outcome_column: str = "revenue"
    date_column: str = "event_date"

    pre_start: str = "2026-01-01"
    pre_end: str = "2026-01-31"

    post_start: str = "2026-02-01"
    post_end: str = "2026-02-28"

    # Statistical assumptions for MDE
    alpha: float = 0.05
    power: float = 0.80


CONFIG = ExperimentConfig()


# ============================================================
# 2. MDE SQL
# ============================================================

def generate_mde_sql(cfg: ExperimentConfig) -> str:
    """
    Produces inputs needed for continuous-outcome MDE calculation.

    For a two-sided test with alpha=.05 and power=.80:

        MDE ≈ (z_alpha + z_power)
              * pooled_SD
              * sqrt(1/N_test + 1/N_control)

    z_alpha ≈ 1.96
    z_power ≈ 0.84
    """

    return dedent(f"""
    -- ========================================================
    -- MDE: TEST VS CONTROL
    -- ========================================================

    WITH customer_outcome AS (

        SELECT
            a.{cfg.customer_id},
            a.{cfg.treatment_column},

            COALESCE(
                SUM(
                    CASE
                        WHEN o.{cfg.date_column}
                            BETWEEN '{cfg.post_start}' AND '{cfg.post_end}'
                        THEN o.{cfg.outcome_column}
                        ELSE 0
                    END
                ),
                0
            ) AS outcome

        FROM {cfg.assignment_table} a

        LEFT JOIN {cfg.outcome_table} o
            ON a.{cfg.customer_id} = o.{cfg.customer_id}

        GROUP BY 1, 2
    ),

    group_stats AS (

        SELECT
            {cfg.treatment_column},
            COUNT(*) AS n,
            AVG(outcome) AS mean_outcome,
            STDDEV_SAMP(outcome) AS sd_outcome

        FROM customer_outcome

        GROUP BY 1
    ),

    stats AS (

        SELECT
            MAX(CASE WHEN {cfg.treatment_column} = 1
                THEN n END) AS n_test,

            MAX(CASE WHEN {cfg.treatment_column} = 0
                THEN n END) AS n_control,

            MAX(CASE WHEN {cfg.treatment_column} = 1
                THEN mean_outcome END) AS test_mean,

            MAX(CASE WHEN {cfg.treatment_column} = 0
                THEN mean_outcome END) AS control_mean,

            MAX(CASE WHEN {cfg.treatment_column} = 1
                THEN sd_outcome END) AS test_sd,

            MAX(CASE WHEN {cfg.treatment_column} = 0
                THEN sd_outcome END) AS control_sd

        FROM group_stats
    )

    SELECT
        *,

        SQRT(
            (
                (n_test - 1) * POWER(test_sd, 2)
                +
                (n_control - 1) * POWER(control_sd, 2)
            )
            /
            NULLIF(n_test + n_control - 2, 0)
        ) AS pooled_sd,

        2.80
        *
        SQRT(
            (
                (n_test - 1) * POWER(test_sd, 2)
                +
                (n_control - 1) * POWER(control_sd, 2)
            )
            /
            NULLIF(n_test + n_control - 2, 0)
        )
        *
        SQRT(
            1.0 / n_test
            +
            1.0 / n_control
        ) AS mde_absolute,

        test_mean - control_mean
            AS observed_difference,

        (test_mean - control_mean)
            / NULLIF(control_mean, 0)
            AS observed_lift_pct

    FROM stats;
    """)


# ============================================================
# 3. TEST / CONTROL BALANCE USING SMD
# ============================================================

def generate_smd_sql(
    cfg: ExperimentConfig,
    covariates: list[str]
) -> str:

    """
    Checks whether Test and Control were balanced BEFORE treatment.

    SMD:

        (Mean_test - Mean_control)
        --------------------------------
        sqrt((Var_test + Var_control) / 2)

    Common interpretation:
        |SMD| < 0.10   -> good balance
        0.10 - 0.20   -> investigate
        > 0.20         -> meaningful imbalance
    """

    queries = []

    for variable in covariates:

        query = f"""
        SELECT

            '{variable}' AS variable,

            AVG(
                CASE WHEN {cfg.treatment_column} = 1
                THEN {variable}
                END
            ) AS test_mean,

            AVG(
                CASE WHEN {cfg.treatment_column} = 0
                THEN {variable}
                END
            ) AS control_mean,

            (
                AVG(
                    CASE WHEN {cfg.treatment_column} = 1
                    THEN {variable}
                    END
                )
                -
                AVG(
                    CASE WHEN {cfg.treatment_column} = 0
                    THEN {variable}
                    END
                )
            )
            /
            NULLIF(
                SQRT(
                    (
                        VAR_SAMP(
                            CASE WHEN {cfg.treatment_column} = 1
                            THEN {variable}
                            END
                        )
                        +
                        VAR_SAMP(
                            CASE WHEN {cfg.treatment_column} = 0
                            THEN {variable}
                            END
                        )
                    ) / 2
                ),
                0
            ) AS smd

        FROM experiment_preperiod
        """

        queries.append(dedent(query).strip())

    return "\n\nUNION ALL\n\n".join(queries) + ";"


# ============================================================
# 4. BUILD PRE/POST CUSTOMER DATASET FOR ANCOVA
# ============================================================

def generate_ancova_dataset_sql(cfg: ExperimentConfig) -> str:

    return dedent(f"""
    -- ========================================================
    -- BUILD CUSTOMER-LEVEL ANCOVA DATASET
    -- ========================================================

    WITH customer_metrics AS (

        SELECT

            a.{cfg.customer_id},

            a.{cfg.treatment_column},

            COALESCE(
                SUM(
                    CASE
                        WHEN o.{cfg.date_column}
                            BETWEEN '{cfg.pre_start}' AND '{cfg.pre_end}'
                        THEN o.{cfg.outcome_column}
                        ELSE 0
                    END
                ),
                0
            ) AS pre_outcome,

            COALESCE(
                SUM(
                    CASE
                        WHEN o.{cfg.date_column}
                            BETWEEN '{cfg.post_start}' AND '{cfg.post_end}'
                        THEN o.{cfg.outcome_column}
                        ELSE 0
                    END
                ),
                0
            ) AS observed_post_outcome

        FROM {cfg.assignment_table} a

        LEFT JOIN {cfg.outcome_table} o
            ON a.{cfg.customer_id} = o.{cfg.customer_id}

        GROUP BY
            1, 2
    )

    SELECT
        {cfg.customer_id},
        {cfg.treatment_column},
        pre_outcome,
        observed_post_outcome

    FROM customer_metrics;
    """)


# ============================================================
# 5. ANCOVA MODEL IN PYTHON
# ============================================================

def run_ancova(df):
    """
    ANCOVA:

        Post Outcome =
            intercept
            + beta1 * Pre Outcome
            + beta2 * Treatment
            + error

    beta2 is the adjusted treatment effect.
    """

    import statsmodels.formula.api as smf

    model = smf.ols(
        formula="""
            observed_post_outcome
            ~ pre_outcome
            + treatment
        """,
        data=df
    ).fit(cov_type="HC3")

    df = df.copy()

    # Actual fitted prediction
    df["predicted_post_outcome"] = model.predict(df)

    # Counterfactual prediction:
    # What would we predict if EVERY customer were Control?
    counterfactual = df.copy()
    counterfactual["treatment"] = 0

    df["predicted_without_treatment"] = model.predict(counterfactual)

    # Customer-level difference relative to predicted baseline
    df["observed_minus_predicted_control"] = (
        df["observed_post_outcome"]
        - df["predicted_without_treatment"]
    )

    return model, df


# ============================================================
# 6. RUN SQL GENERATOR
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("1. MDE SQL")
    print("=" * 70)
    print(generate_mde_sql(CONFIG))

    print("\n")
    print("=" * 70)
    print("2. BALANCE / SMD SQL")
    print("=" * 70)

    covariates = [
        "pre_revenue",
        "pre_orders",
        "pre_sessions",
        "customer_tenure"
    ]

    print(generate_smd_sql(CONFIG, covariates))

    print("\n")
    print("=" * 70)
    print("3. ANCOVA DATASET SQL")
    print("=" * 70)
    print(generate_ancova_dataset_sql(CONFIG))
