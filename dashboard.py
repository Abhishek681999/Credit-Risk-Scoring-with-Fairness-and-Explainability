"""Interactive dashboard for the credit-risk portfolio project.
Run from the project folder:
    streamlit run dashboard.py
"""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pipeline_utils import (
    load_settings,
    logit_feature,
    prepare,
    remove_kaggle_index,
)
from project_config import AGE_GROUPS, OUTPUTS, TARGET
ROOT = Path(__file__).resolve().parent
MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "ann",
)
RAW_FIELDS = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)
OPTIONAL_MISSING = {
    "MonthlyIncome",
    "NumberOfDependents",
}
COUNT_FIELDS = {
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
}
st.set_page_config(
    page_title="Credit Risk | Portfolio Analytics",
    page_icon="📊",
    layout="wide",
)
@st.cache_data
def read_table(path_string):
    return pd.read_csv(path_string)
@st.cache_data
def feature_list():
    path = OUTPUTS / "models" / "features.json"
    return json.loads(path.read_text(encoding="utf-8"))
@st.cache_resource
def load_model_and_calibrator(model_name):
    model_path = OUTPUTS / "models" / f"{model_name}.joblib"
    calibrator_path = (
        OUTPUTS / "evaluation" / f"{model_name}_calibrator.joblib"
    )
    if not model_path.exists() or not calibrator_path.exists():
        raise FileNotFoundError(
            f"Missing model or calibrator for {model_name}. "
            "Run 03_train_models.py and 04_evaluate_policies.py."
        )
    return (
        joblib.load(model_path),
        joblib.load(calibrator_path),
    )
def validate_inputs(frame):
    """Validate the original applicant fields before preprocessing."""
    missing_columns = [
        column for column in RAW_FIELDS
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise ValueError("The CSV has no applicant rows.")
    if len(frame) > 200_000:
        raise ValueError(
            "Upload at most 200,000 rows at a time."
        )
    clean = frame.copy()
    for column in RAW_FIELDS:
        original = clean[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_text = original.notna() & converted.isna()
        if invalid_text.any():
            raise ValueError(
                f"{column} has non-numeric values in "
                f"{int(invalid_text.sum())} row(s)."
            )
        clean[column] = converted
        if (
            column not in OPTIONAL_MISSING
            and converted.isna().any()
        ):
            raise ValueError(
                f"{column} has missing values."
            )
        non_missing = converted.dropna()
        if not np.isfinite(non_missing.to_numpy()).all():
            raise ValueError(
                f"{column} contains infinity."
            )
        if (non_missing < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )
        if (
            column in COUNT_FIELDS
            and (non_missing % 1 != 0).any()
        ):
            raise ValueError(
                f"{column} must contain whole numbers."
            )
    if (clean["age"] <= 0).any():
        raise ValueError(
            "Age must be greater than zero."
        )
    return clean
def score_applicants(frame, model_name):
    """Apply saved preprocessing, model, and calibration."""
    clean = validate_inputs(frame)
    processed = prepare(clean, load_settings())
    features = feature_list()
    X = processed[features]
    if X.isna().any().any():
        raise ValueError(
            "Missing model inputs remain after preprocessing."
        )
    model, calibrator = load_model_and_calibrator(model_name)
    raw_scores = model.predict_proba(X)[:, 1]
    calibrated_scores = calibrator.predict_proba(
        logit_feature(raw_scores)
    )[:, 1]
    if not np.isfinite(calibrated_scores).all():
        raise ValueError(
            "The model returned invalid scores."
        )
    return (
        raw_scores,
        calibrated_scores,
        processed["age_group"].to_numpy(),
    )
def decision_metrics(y, scores, threshold, fn_cost):
    flagged = scores >= threshold
    false_positives = int(
        np.sum((y == 0) & flagged)
    )
    false_negatives = int(
        np.sum((y == 1) & ~flagged)
    )
    true_positives = int(
        np.sum((y == 1) & flagged)
    )
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "flag_rate": float(flagged.mean()),
        "cost_per_1000": (
            1000
            * (
                false_positives
                + fn_cost * false_negatives
            )
            / len(y)
        ),
    }
def age_metrics(y, scores, groups, threshold):
    flagged = scores >= threshold
    rows = []
    for age_group in AGE_GROUPS:
        mask = groups == age_group
        actual = y[mask]
        predicted = flagged[mask]
        negatives = actual == 0
        positives = actual == 1
        rows.append(
            {
                "age_group": age_group,
                "applicants": int(mask.sum()),
                "observed_default_rate": float(
                    actual.mean()
                ),
                "flag_rate": float(
                    predicted.mean()
                ),
                "false_positive_rate": float(
                    predicted[negatives].mean()
                ),
                "true_positive_rate": float(
                    predicted[positives].mean()
                ),
            }
        )
    table = pd.DataFrame(rows)
    dp_difference = (
        table["flag_rate"].max()
        - table["flag_rate"].min()
    )
    eo_difference = max(
        table["false_positive_rate"].max()
        - table["false_positive_rate"].min(),
        table["true_positive_rate"].max()
        - table["true_positive_rate"].min(),
    )
    return table, dp_difference, eo_difference
def show_image(path, caption=None):
    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(f"Chart not found: {path}")
st.sidebar.radio(
    "Portfolio",
    ["Borrower delinquency", "Consumer loans"],
    key="active_portfolio",
)
st.sidebar.divider()

if st.session_state["active_portfolio"] == "Borrower delinquency":
    # Check that the important files exist before drawing the app.
    required_files = [
        OUTPUTS / "evaluation" / "all5_summary.csv",
        OUTPUTS
        / "evaluation"
        / "calibrated_validation_predictions.csv",
        OUTPUTS / "models" / "features.json",
        ROOT
        / "data"
        / "processed"
        / "preprocessing_settings.json",
    ]
    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]
    if missing_files:
        st.error(
            "Project outputs are missing. Run the training "
            "and evaluation scripts first:\n\n"
            + "\n".join(missing_files)
        )
        st.stop()
    summary = read_table(
        str(OUTPUTS / "evaluation" / "all5_summary.csv")
    )
    validation_scores = read_table(
        str(
            OUTPUTS
            / "evaluation"
            / "calibrated_validation_predictions.csv"
        )
    )
    y_val = validation_scores["y_true"].to_numpy()
    groups_val = validation_scores["age_group"].to_numpy()
    # ------------------------------------------------------------------
    # Header and decision controls
    # ------------------------------------------------------------------
    st.title("Borrower Credit Risk | Fairness Audit")
    st.caption(
        "Give Me Some Credit: two-year borrower delinquency. "
        "Scores and review flags are historical demonstrations, not lending decisions."
    )
    st.sidebar.header("Borrower-risk settings")
    st.sidebar.caption(
        "These controls apply to the Give Me Some Credit tabs only. "
        "The consumer-loan tab has separate controls."
    )
    selected_model = st.sidebar.selectbox(
        "Model",
        MODELS,
        index=MODELS.index("lightgbm"),
    )
    fn_cost = st.sidebar.slider(
        "Cost of missed default, relative to false alarm",
        min_value=1,
        max_value=30,
        value=10,
    )
    theoretical_threshold = 1 / (fn_cost + 1)
    manual_threshold = st.sidebar.checkbox(
        "Set threshold manually",
        value=False,
    )
    if manual_threshold:
        threshold = st.sidebar.slider(
            "Default probability cutoff",
            min_value=0.01,
            max_value=0.50,
            value=float(round(theoretical_threshold, 3)),
            step=0.001,
        )
    else:
        threshold = theoretical_threshold
    st.sidebar.write(
        f"Current cutoff: **{threshold:.2%}**"
    )
    st.sidebar.caption(
        "The cost ratio is illustrative. No real lending "
        "costs were supplied with this dataset."
    )
    tabs = st.tabs(
        [
            "Overview",
            "Data",
            "Models",
            "Policy & fairness",
            "Explainability",
            "Calibration",
            "Score an applicant",
        ]
    )
    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("What this project tests")
        a, b, c = st.columns(3)
        a.metric(
            "Training applicants",
            "119,999",
        )
        b.metric(
            "Validation applicants",
            f"{len(y_val):,}",
        )
        c.metric(
            "Observed default rate",
            f"{y_val.mean():.2%}",
        )
        st.write(
            "The models rank two-year default risk. We calibrate "
            "their probabilities, apply a cost-based review cutoff, "
            "compare errors across age groups, and inspect "
            "individual explanations."
        )
        st.info(
            "Provisional model: LightGBM. It narrowly led on "
            "validation average precision and illustrative 10:1 cost. "
            "XGBoost was very close; the preferred model can change "
            "with cost assumptions."
        )
        st.caption(
            "Saved comparison below: FN:FP cost 10:1 and cutoff 9.09%. "
            "Sidebar settings affect the interactive Policy & fairness "
            "tab and applicant scoring."
        )
        st.dataframe(
            summary[
                [
                    "model",
                    "roc_auc",
                    "average_precision",
                    "calibrated_brier",
                    "cost_per_1000",
                    "dp_difference",
                    "eo_difference",
                ]
            ].round(4),
            hide_index=True,
            use_container_width=True,
        )
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader(
            "Data quality and observed outcomes"
        )
        raw_path = ROOT / "data" / "cs-training.csv"
        if raw_path.exists():
            raw = remove_kaggle_index(
                read_table(str(raw_path))
            )
            a, b, c = st.columns(3)
            a.metric(
                "Rows",
                f"{len(raw):,}",
            )
            b.metric(
                "Missing MonthlyIncome",
                f"{raw['MonthlyIncome'].isna().mean():.2%}",
            )
            c.metric(
                "Missing dependents",
                f"{raw['NumberOfDependents'].isna().mean():.2%}",
            )
            st.write(
                f"Exact duplicate rows retained: "
                f"**{raw.duplicated().sum():,}**. "
                f"Invalid ages of zero or less removed "
                f"before training: "
                f"**{(raw['age'] <= 0).sum():,}**."
            )
            counts = (
                raw[TARGET]
                .value_counts()
                .sort_index()
                .rename(
                    index={
                        0: "No default",
                        1: "Default",
                    }
                )
            )
            st.bar_chart(counts)
        else:
            st.info(
                "Place cs-training.csv in data/ to show "
                "raw-data counts. Saved EDA charts are below."
            )
        for filename, title in (
            ("class_balance.png", "Class balance"),
            (
                "observed_default_by_age.png",
                "Observed defaults by age",
            ),
            (
                "feature_distributions.png",
                "Feature distributions",
            ),
            (
                "correlations.png",
                "Correlation heatmap",
            ),
        ):
            with st.expander(title):
                show_image(
                    OUTPUTS / "eda" / filename
                )
    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader(
            "Five models on the same 30,000 validation rows"
        )
        st.write(
            "ROC-AUC measures ranking across thresholds. "
            "Average precision is useful when defaults are "
            "uncommon. Brier score describes probability quality; "
            "lower is better."
        )
        st.caption(
            "The saved false-positive, false-negative, and cost "
            "columns use FN:FP cost 10:1 and cutoff 9.09%. "
            "Change the sidebar controls to explore another policy "
            "in the Policy & fairness tab."
        )
        columns = [
            "model",
            "roc_auc",
            "average_precision",
            "raw_brier",
            "calibrated_brier",
            "false_positives",
            "false_negatives",
            "cost_per_1000",
        ]
        st.dataframe(
            summary[columns].round(4),
            hide_index=True,
            use_container_width=True,
        )
        # Side-by-side bars: ROC-AUC and average precision
        # must not be added together.
        st.bar_chart(
            summary.set_index("model")[
                ["roc_auc", "average_precision"]
            ],
            stack=False,
        )
        st.caption(
            "Raw Brier scores are affected by class weighting. "
            "Use calibrated scores for the probability-based policy."
        )
    # ------------------------------------------------------------------
    # Interactive policy and fairness
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader(
            "Change the cutoff and watch decisions change"
        )
        score_column = f"{selected_model}_calibrated"
        scores = validation_scores[
            score_column
        ].to_numpy()
        results = decision_metrics(
            y_val,
            scores,
            threshold,
            fn_cost,
        )
        a, b, c, d = st.columns(4)
        a.metric(
            "Flagged for review",
            f"{results['flag_rate']:.2%}",
        )
        b.metric(
            "False positives",
            f"{results['false_positives']:,}",
        )
        c.metric(
            "Missed defaults",
            f"{results['false_negatives']:,}",
        )
        d.metric(
            "Illustrative cost / 1,000",
            f"{results['cost_per_1000']:.1f}",
        )
        st.write(
            "Lowering the cutoff generally catches more "
            "defaults but flags more applicants who "
            "would not default."
        )
        # Recompute costs using the current sidebar cost ratio.
        cutoffs = np.linspace(0.01, 0.40, 80)
        cost_curve = [
            decision_metrics(
                y_val,
                scores,
                cutoff,
                fn_cost,
            )["cost_per_1000"]
            for cutoff in cutoffs
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            cutoffs,
            cost_curve,
            color="#1769aa",
        )
        ax.axvline(
            threshold,
            color="#d1495b",
            linestyle="--",
            label=f"Selected: {threshold:.2%}",
        )
        ax.set(
            xlabel="Calibrated default-probability cutoff",
            ylabel="Illustrative cost per 1,000",
            title=f"{selected_model}: cost versus cutoff",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader(
            "Age-group audit at this cutoff"
        )
        group_table, dp_gap, eo_gap = age_metrics(
            y_val,
            scores,
            groups_val,
            threshold,
        )
        left, right = st.columns(2)
        left.metric(
            "Demographic parity difference",
            f"{dp_gap:.2%}",
        )
        right.metric(
            "Equalized odds difference",
            f"{eo_gap:.2%}",
        )
        display_groups = group_table.copy()
        rate_columns = (
            "observed_default_rate",
            "flag_rate",
            "false_positive_rate",
            "true_positive_rate",
        )
        for column in rate_columns:
            display_groups[column] *= 100
        st.dataframe(
            display_groups.round(2),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Rate columns are percentages. Age is excluded "
            "from model inputs and retained for the audit. "
            "The under-25 validation group contains only "
            "427 applicants."
        )
        st.subheader(
            "Equalized-odds mitigation experiment"
        )
        mitigation_path = (
            OUTPUTS / "mitigation" / "comparison.csv"
        )
        if mitigation_path.exists():
            mitigation = read_table(
                str(mitigation_path)
            )
            experiment_model = st.selectbox(
                "Mitigation experiment model",
                ("xgboost", "lightgbm", "ann"),
            )
            selected_rows = mitigation[
                mitigation["model"] == experiment_model
            ]
            st.dataframe(
                selected_rows[
                    [
                        "method",
                        "false_positives",
                        "false_negatives",
                        "cost_per_1000",
                        "dp_difference",
                        "eo_difference",
                    ]
                ].round(4),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "This experiment fits mitigation on "
                "15,000 validation rows and evaluates it on "
                "another 15,000. Compare mitigated rows with "
                "the baseline row in this experiment. "
                "The table does not respond to the sidebar cutoff."
            )
        else:
            st.info(
                "Run 06_mitigation.py to show "
                "mitigation results."
            )
    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader(
            "Which inputs influenced model scores?"
        )
        explained_model = st.selectbox(
            "Tree model to explain",
            ("lightgbm", "xgboost"),
        )
        show_image(
            OUTPUTS
            / "explainability"
            / f"{explained_model}_shap_beeswarm.png",
            caption=(
                "Each dot represents an applicant. "
                "Right pushes the raw model score higher; "
                "left pushes it lower."
            ),
        )
        example = st.selectbox(
            "Individual validation example",
            (
                "true_positive",
                "false_positive",
                "false_negative",
            ),
        )
        show_image(
            OUTPUTS
            / "explainability"
            / f"{explained_model}_{example}_waterfall.png",
        )
        st.caption(
            "Case names were chosen using LightGBM decisions. "
            "For example, LightGBM's 'false_negative' "
            "may be flagged by XGBoost. SHAP values explain "
            "raw model scores, not percentage-point changes "
            "in calibrated probability."
        )
        permutation_path = (
            OUTPUTS
            / "explainability"
            / "five_model_permutation_importance.csv"
        )
        if permutation_path.exists():
            importance = read_table(
                str(permutation_path)
            )
            chosen = (
                importance[
                    importance["model"] == selected_model
                ]
                .sort_values(
                    "ap_drop_mean",
                    ascending=False,
                )
                .head(10)
            )
            st.write(
                f"Permutation importance for "
                f"**{selected_model}**: loss in average "
                "precision after shuffling a feature."
            )
            st.bar_chart(
                chosen.set_index("feature")[
                    "ap_drop_mean"
                ]
            )
    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    with tabs[5]:
        st.subheader(
            "Are predicted probabilities believable?"
        )
        overall_path = (
            OUTPUTS
            / "reliability"
            / "overall_reliability.csv"
        )
        age_path = (
            OUTPUTS
            / "reliability"
            / "reliability_by_age.csv"
        )
        if overall_path.exists():
            st.dataframe(
                read_table(
                    str(overall_path)
                ).round(4),
                hide_index=True,
                use_container_width=True,
            )
        show_image(
            OUTPUTS
            / "reliability"
            / "reliability_curves.png",
            caption=(
                "Points nearer the diagonal show better "
                "agreement between predicted probabilities "
                "and observed default rates."
            ),
        )
        if age_path.exists():
            st.write(
                "Average predictions and outcomes "
                "by age group"
            )
            st.dataframe(
                read_table(
                    str(age_path)
                ).round(4),
                hide_index=True,
                use_container_width=True,
            )
        st.caption(
            "Overall calibration can look good even "
            "when age groups differ. Estimates for "
            "small groups are less stable."
        )
    # ------------------------------------------------------------------
    # Individual and batch scoring
    # ------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Score an applicant")
        st.write(
            "Enter the ten fields used by Give Me Some Credit. "
            "The dashboard applies the saved training "
            "preprocessing and calibration settings. "
            "Age determines the audit group but is not "
            "directly used as a model feature."
        )
        with st.form("applicant_form"):
            col1, col2 = st.columns(2)
            with col1:
                utilization = st.number_input(
                    "Revolving credit utilization",
                    min_value=0.0,
                    value=0.20,
                    step=0.01,
                )
                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=40,
                    step=1,
                )
                late_30_59 = st.number_input(
                    "Times 30–59 days late",
                    min_value=0,
                    value=0,
                    step=1,
                )
                debt_ratio = st.number_input(
                    "Debt ratio",
                    min_value=0.0,
                    value=0.35,
                    step=0.01,
                )
                income_missing = st.checkbox(
                    "Monthly income unknown"
                )
                monthly_income = st.number_input(
                    "Monthly income",
                    min_value=0.0,
                    value=5400.0,
                    step=100.0,
                    disabled=income_missing,
                )
            with col2:
                open_lines = st.number_input(
                    "Open credit lines and loans",
                    min_value=0,
                    value=5,
                    step=1,
                )
                late_90 = st.number_input(
                    "Times 90 days late",
                    min_value=0,
                    value=0,
                    step=1,
                )
                real_estate = st.number_input(
                    "Real estate loans or lines",
                    min_value=0,
                    value=1,
                    step=1,
                )
                late_60_89 = st.number_input(
                    "Times 60–89 days late",
                    min_value=0,
                    value=0,
                    step=1,
                )
                dependents_missing = st.checkbox(
                    "Number of dependents unknown"
                )
                dependents = st.number_input(
                    "Number of dependents",
                    min_value=0,
                    value=0,
                    step=1,
                    disabled=dependents_missing,
                )
            submitted = st.form_submit_button(
                "Calculate default-risk estimate"
            )
        if submitted:
            st.session_state["applicant_input"] = {
                "RevolvingUtilizationOfUnsecuredLines":
                    utilization,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse":
                    late_30_59,
                "DebtRatio": debt_ratio,
                "MonthlyIncome": (
                    np.nan
                    if income_missing
                    else monthly_income
                ),
                "NumberOfOpenCreditLinesAndLoans":
                    open_lines,
                "NumberOfTimes90DaysLate":
                    late_90,
                "NumberRealEstateLoansOrLines":
                    real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse":
                    late_60_89,
                "NumberOfDependents": (
                    np.nan
                    if dependents_missing
                    else dependents
                ),
            }
        if "applicant_input" in st.session_state:
            try:
                applicant = pd.DataFrame(
                    [st.session_state["applicant_input"]]
                )
                raw_score, calibrated, age_group = (
                    score_applicants(
                        applicant,
                        selected_model,
                    )
                )
                score = float(calibrated[0])
                st.metric(
                    f"{selected_model} calibrated "
                    "default probability",
                    f"{score:.2%}",
                )
                st.write(
                    f"Age audit group: "
                    f"**{age_group[0]}**"
                )
                st.write(
                    f"Current review cutoff: "
                    f"**{threshold:.2%}**"
                )
                if score >= threshold:
                    st.warning(
                        "Flagged for review under this "
                        "illustrative policy."
                    )
                else:
                    st.success(
                        "Not flagged for review under "
                        "this illustrative policy."
                    )
                with st.expander(
                    "Why is the raw model score different?"
                ):
                    st.write(
                        f"Raw model probability: "
                        f"{raw_score[0]:.2%}. "
                        "The review decision uses the "
                        "calibrated probability above."
                    )
            except (
                ValueError,
                FileNotFoundError,
                KeyError,
            ) as error:
                st.error(str(error))
        st.divider()
        st.subheader(
            "Score a CSV of matching applicants"
        )
        st.write(
            "Upload `cs-test.csv` or another CSV with "
            "the same ten input columns. MonthlyIncome "
            "and NumberOfDependents may be blank. "
            "Extra ID or target columns are not used "
            "by the model."
        )
        with st.form("batch_form"):
            upload = st.file_uploader(
                "Applicant CSV",
                type=["csv"],
            )
            batch_submitted = (
                st.form_submit_button(
                    "Score uploaded CSV"
                )
            )
        if batch_submitted:
            if upload is None:
                st.error(
                    "Choose a CSV file first."
                )
            else:
                try:
                    frame = pd.read_csv(upload)
                    _, calibrated, age_groups = (
                        score_applicants(
                            frame,
                            selected_model,
                        )
                    )
                    if "Id" in frame.columns:
                        ids = frame["Id"].to_numpy()
                    elif "Unnamed: 0" in frame.columns:
                        ids = frame[
                            "Unnamed: 0"
                        ].to_numpy()
                    else:
                        ids = np.arange(
                            1,
                            len(frame) + 1,
                        )
                    st.session_state[
                        "batch_scores"
                    ] = pd.DataFrame(
                        {
                            "Id": ids,
                            "calibrated_default_probability":
                                calibrated,
                            "age_group": age_groups,
                        }
                    )
                    st.session_state[
                        "batch_model"
                    ] = selected_model
                except (
                    ValueError,
                    FileNotFoundError,
                    KeyError,
                ) as error:
                    st.error(str(error))
        if "batch_scores" in st.session_state:
            batch = st.session_state[
                "batch_scores"
            ].copy()
            batch["flagged_for_review"] = (
                batch[
                    "calibrated_default_probability"
                ] >= threshold
            )
            st.write(
                f"**{len(batch):,} rows** scored with "
                f"**{st.session_state['batch_model']}**. "
                "The flag column follows the current cutoff."
            )
            st.dataframe(
                batch.head(100),
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download scored CSV",
                data=batch.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "credit_risk_scored_applicants.csv"
                ),
                mime="text/csv",
            )
        st.caption(
            "This is a demonstration based on historical "
            "competition data. The scores and review flags "
            "are not validated for live lending decisions."
        )



# ------------------------------------------------------------------
# Independent historical consumer-loan portfolio
# ------------------------------------------------------------------
CONSUMER_OUTPUTS = OUTPUTS / "consumer_loans"
CONSUMER_VALIDATION = ROOT / "data" / "processed" / "consumer_validation_raw.csv"
CONSUMER_NUMERIC = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "pub_rec_bankruptcies",
    "credit_history_years",
)
CONSUMER_CATEGORICAL = (
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
)
CONSUMER_RAW_FIELDS = (
    tuple(field for field in CONSUMER_NUMERIC if field != "credit_history_years")
    + CONSUMER_CATEGORICAL
    + ("earliest_cr_line", "issue_year")
)


def prepare_consumer_example(frame):
    """Use the same feature definitions as 13_train_consumer_models.py."""
    missing = [field for field in CONSUMER_RAW_FIELDS if field not in frame]
    if missing:
        raise ValueError("Missing consumer-loan columns: " + ", ".join(missing))
    example = frame.copy()
    issue_year = pd.to_numeric(example["issue_year"], errors="coerce")
    if issue_year.isna().any() or not (issue_year == 2014).all():
        raise ValueError("The example must describe a 2014 loan.")
    earliest = pd.to_datetime(
        example["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    if earliest.isna().any():
        raise ValueError("earliest_cr_line must look like Jan-2000.")
    example["credit_history_years"] = issue_year - earliest.dt.year
    if (example["credit_history_years"] < 0).any():
        raise ValueError("Earliest credit line cannot be after loan issuance.")
    for field in CONSUMER_NUMERIC:
        original = example[field]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            raise ValueError(f"{field} must be numeric.")
        if (converted.dropna() < 0).any():
            raise ValueError(f"{field} cannot be negative.")
        example[field] = converted
    return example[list(CONSUMER_NUMERIC + CONSUMER_CATEGORICAL)]


@st.cache_resource
def load_consumer_model(model_name):
    return joblib.load(CONSUMER_OUTPUTS / f"{model_name}.joblib")



if st.session_state["active_portfolio"] == "Consumer loans":
    st.sidebar.header("Consumer-loan controls")
    consumer_model = st.sidebar.selectbox(
        "Model for review ordering",
        ["lightgbm", "xgboost", "logistic_regression"],
        key="consumer_review_model",
    )
    review_share = st.sidebar.slider(
        "Share of 2014 loans sent to review",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        format="%d%%",
        key="consumer_review_share",
    )
    st.sidebar.caption(
        "These controls affect the review queue and risk-decile chart "
        "in the Consumer loans study only."
    )

    st.title("Consumer Loan Risk | historical pilot")
    st.caption(
        "Lending Club 36-month loans. Target: eventual charge-off versus "
        "fully paid. Train: 2012–2013; later-year evaluation: 2014."
    )
    consumer_tabs = st.tabs(
        ["Overview", "Models", "Review queue", "Reliability", "Drivers", "Try a loan"]
    )
    required_consumer = {
        "model comparison": CONSUMER_OUTPUTS / "model_comparison.csv",
        "calibration": CONSUMER_OUTPUTS / "calibration_comparison.csv",
        "ranked loans": CONSUMER_OUTPUTS / "validation_predictions.csv",
        "loan purpose": CONSUMER_OUTPUTS / "purpose_outcomes.csv",
        "feature importance": CONSUMER_OUTPUTS / "consumer_feature_importance.csv",
        "validation loans": CONSUMER_VALIDATION,
        "LightGBM model": CONSUMER_OUTPUTS / "lightgbm.joblib",
        "XGBoost model": CONSUMER_OUTPUTS / "xgboost.joblib",
        "logistic model": CONSUMER_OUTPUTS / "logistic_regression.joblib",
    }
    absent = [name for name, path in required_consumer.items() if not path.exists()]
    if absent:
        st.warning(
            "Consumer-loan outputs are missing: "
            + ", ".join(absent)
            + ". Run scripts 12 through 17 from the project folder."
        )
        st.stop()

    model_results = read_table(str(required_consumer["model comparison"]))
    calibration_results = read_table(str(required_consumer["calibration"]))
    purposes = read_table(str(required_consumer["loan purpose"]))
    importance = read_table(str(required_consumer["feature importance"]))
    loan_predictions = read_table(str(required_consumer["ranked loans"]))
    loan_validation = read_table(str(required_consumer["validation loans"]))
    if (
        len(loan_predictions) != len(loan_validation)
        or not np.array_equal(
            loan_predictions["y_true"].to_numpy(),
            loan_validation["target"].to_numpy(),
        )
    ):
        st.error("Consumer-loan predictions are not aligned with validation.")
        st.stop()

    observed = loan_predictions["y_true"].to_numpy()
    scores = loan_predictions[f"{consumer_model}_probability"].to_numpy()
    n_review = int(round(len(scores) * review_share / 100))
    highest = np.argsort(-scores, kind="stable")[:n_review]
    captured = int(observed[highest].sum())
    captured_share = captured / observed.sum()
    ranked = pd.DataFrame(
        {
            "score": scores,
            "charged_off": observed,
            "risk_decile": pd.qcut(
                pd.Series(scores).rank(method="first"), 10, labels=False
            )
            + 1,
        }
    )
    decile_table = ranked.groupby("risk_decile", as_index=False).agg(
        loans=("charged_off", "size"),
        observed_charge_off_rate=("charged_off", "mean"),
        mean_predicted_probability=("score", "mean"),
    )

    with consumer_tabs[0]:
        st.subheader("The question and the data")
        st.write(
            "Can information recorded when a 36-month consumer loan was issued "
            "help rank its eventual charge-off risk? Only completed 2012–2014 "
            "loans are included. Current loans are not called successful loans."
        )
        a, b, c = st.columns(3)
        a.metric("Training loans | 2012–13", "143,892")
        b.metric("Later-year loans | 2014", f"{len(observed):,}")
        c.metric("2014 charge-off rate", f"{observed.mean():.2%}")
        st.info(
            "This data ends in 2018; evaluation here uses 2014 loans. "
            "The model is a historical portfolio demonstration, not a "
            "validated 2026 decision system. Age is unavailable, so the "
            "other study's age fairness audit cannot be repeated here."
        )
        st.write(
            "The model separates some risk, but only modestly. At the "
            "selected review capacity, the queue contains "
            f"**{captured_share:.2%}** of the observed 2014 charge-offs. "
            "The review queue tab shows the trade-off."
        )

    with consumer_tabs[1]:
        st.subheader("Three models on the same 2014 cohort")
        st.dataframe(
            model_results.round(4), hide_index=True, use_container_width=True
        )
        st.write(
            "Logistic regression is the interpretable baseline. LightGBM "
            "and XGBoost have nearly identical ranking results; LightGBM "
            "has a marginally lower Brier score. The models are separate "
            "from those in the borrower-delinquency study."
        )
        show_image(
            CONSUMER_OUTPUTS / "precision_recall.png",
            "Precision–recall on the 2014 consumer-loan cohort.",
        )
        st.caption(
            "The 2014 average-precision reference is its 13.73% "
            "charge-off prevalence. These results do not justify "
            "automatic approval or rejection."
        )

    with consumer_tabs[2]:
        st.subheader("Review capacity and risk concentration")
        st.write(
            f"Sidebar settings: **{consumer_model}**, with the highest-risk "
            f"**{review_share}%** of loans placed in a hypothetical review queue."
        )
        a, b, c = st.columns(3)
        a.metric("Loans in queue", f"{n_review:,}")
        b.metric("Charge-offs in queue", f"{captured:,}")
        c.metric("Share of charge-offs captured", f"{captured_share:.2%}")
        st.caption(
            "This is retrospective: being placed in a queue does not mean "
            "review would have prevented a charge-off."
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(
            decile_table["risk_decile"],
            decile_table["observed_charge_off_rate"] * 100,
        )
        ax.axhline(
            observed.mean() * 100,
            color="red",
            linestyle="--",
            label="2014 portfolio average",
        )
        ax.set(
            xlabel="Predicted-risk decile (10 = highest)",
            ylabel="Observed charge-off rate (%)",
            title=f"2014 outcomes ranked by {consumer_model}",
        )
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Risk-decile counts and predicted probabilities"):
            st.dataframe(
                decile_table.round(4), hide_index=True, use_container_width=True
            )
        st.subheader("Loan-purpose outcomes")
        top_purposes = purposes.nlargest(8, "validation_loans").iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(
            top_purposes["purpose"],
            top_purposes["validation_charge_off_rate"] * 100,
        )
        ax.set(
            xlabel="Observed charge-off rate (%)",
            title="2014 outcomes by loan purpose",
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Purpose comparisons are historical associations, not "
            "evidence that a purpose caused charge-off."
        )
        with st.expander("Purpose table, 2012–13 versus 2014"):
            st.dataframe(
                purposes.head(12).round(4),
                hide_index=True,
                use_container_width=True,
            )

    with consumer_tabs[3]:
        st.subheader("Probability reliability over time")
        st.write(
            "2014 charge-offs occurred in 13.73% of loans. LightGBM's "
            "average prediction was 12.61%. Sigmoid calibration fitted "
            "using 2012–2013 data did not remove the later-year gap."
        )
        st.dataframe(
            calibration_results.round(4),
            hide_index=True,
            use_container_width=True,
        )
        show_image(
            CONSUMER_OUTPUTS / "reliability_by_decile.png",
            "All three raw models: predicted probability versus observed rate.",
        )
        st.caption(
            "The chart shows raw probabilities for all three models; "
            "the table above includes calibrated LightGBM and XGBoost."
        )

    with consumer_tabs[4]:
        st.subheader("Feature importance | provisional LightGBM")
        st.write(
            "Shuffling one feature at a time in a stratified sample of "
            "10,000 later-year loans measures its effect on ranking. "
            "This does not give the direction of an effect or a causal reason."
        )
        show_image(
            CONSUMER_OUTPUTS / "consumer_feature_importance.png",
            "LightGBM permutation importance on a 2014 validation sample.",
        )
        st.dataframe(
            importance.round(4), hide_index=True, use_container_width=True
        )
        st.caption(
            "This tab explains LightGBM regardless of the sidebar's "
            "review-ordering model selection."
        )

    with consumer_tabs[5]:
        st.subheader("Try a historical 2014 loan")
        st.write(
            "Choose a row, edit its origination fields and obtain a score "
            "from the saved model. This is not a current loan offer or "
            "an automatic approve/decline decision."
        )
        scoring_model = st.selectbox(
            "Model for this example",
            ["lightgbm", "xgboost", "logistic_regression"],
            key="consumer_scoring_model",
        )
        example_number = st.number_input(
            "2014 validation row (starting at zero)",
            min_value=0,
            max_value=len(loan_validation) - 1,
            value=0,
            step=1,
            key="consumer_example_number",
        )
        example = loan_validation.iloc[[int(example_number)]][
            list(CONSUMER_RAW_FIELDS)
        ].copy().reset_index(drop=True)
        edited = st.data_editor(
            example,
            disabled=["issue_year"],
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            key=f"consumer_editor_{example_number}",
        )
        if st.button("Score illustrative loan", key="consumer_score_button"):
            try:
                inputs = prepare_consumer_example(edited)
                probability = float(
                    load_consumer_model(scoring_model).predict_proba(inputs)[0, 1]
                )
                percentile = 100 * np.mean(
                    loan_predictions[f"{scoring_model}_probability"].to_numpy()
                    <= probability
                )
                st.metric("Historical model charge-off score", f"{probability:.2%}")
                st.write(
                    f"This score is above approximately {percentile:.1f}% "
                    "of the saved 2014 scores from the same model."
                )
                st.caption(
                    "Probabilities were generally below actual 2014 "
                    "charge-off rates. Use the Reliability tab to interpret "
                    "this score's limitation."
                )
            except (ValueError, KeyError, TypeError) as error:
                st.error(str(error))
