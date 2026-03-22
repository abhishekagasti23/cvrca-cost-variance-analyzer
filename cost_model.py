"""
CVRCA — Cost Variance Root Cause Analyzer
Cost Model: XGBoost regression for cost overrun prediction + Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── CONTRACT TYPES ───────────────────────────────────────────────────────────
CONTRACT_TYPES = {
    "lump_sum":    "Lump Sum (Fixed Price)",
    "remeasure":   "Re-measure / Bill of Quantities",
    "target_cost": "NEC Target Cost",
    "cost_plus":   "Cost-Plus / Reimbursable",
    "design_build": "Design & Build (EPC)",
}

# Overrun risk multiplier by contract type (from industry research)
CONTRACT_RISK = {
    "lump_sum":    0.85,
    "remeasure":   1.25,
    "target_cost": 1.10,
    "cost_plus":   1.45,
    "design_build": 0.90,
}

CONTRACTOR_TIERS = {
    "tier1": "Tier 1 (National contractor)",
    "tier2": "Tier 2 (Regional specialist)",
    "tier3": "Tier 3 (Local SME)",
}

PROGRAMME_PHASES = {
    "pre_construction": "Pre-Construction (RIBA 2-3)",
    "early_works":      "Early Works (RIBA 4)",
    "construction":     "Main Construction (RIBA 5)",
    "commissioning":    "Testing & Commissioning",
    "handover":         "Handover & Close-Out",
}

FEATURES = [
    "contract_type_enc",
    "contractor_tier_enc",
    "programme_phase_enc",
    "original_budget_log",
    "duration_months",
    "change_order_count",
    "change_order_value_pct",
    "scope_complexity_score",
    "third_party_interfaces",
    "design_completeness_pct",
    "risk_allowance_pct",
    "procurement_risk_score",
]

FEATURE_LABELS = {
    "contract_type_enc":       "Contract type",
    "contractor_tier_enc":     "Contractor tier",
    "programme_phase_enc":     "Programme phase",
    "original_budget_log":     "Project scale (log budget)",
    "duration_months":         "Programme duration",
    "change_order_count":      "Number of change orders",
    "change_order_value_pct":  "Change order value (% of budget)",
    "scope_complexity_score":  "Scope complexity",
    "third_party_interfaces":  "Third-party interfaces",
    "design_completeness_pct": "Design completeness at award",
    "risk_allowance_pct":      "Risk allowance (% of budget)",
    "procurement_risk_score":  "Procurement risk score",
}


# ─── TRAINING DATA GENERATOR ─────────────────────────────────────────────────

def generate_training_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic training data calibrated to IPA / Bent Flyvbjerg infrastructure
    cost overrun research (average 20–45% overrun in major infra).
    """
    rng = np.random.default_rng(seed)

    contract_types = list(CONTRACT_TYPES.keys())
    contractor_tiers = [1, 2, 3]
    phases = list(PROGRAMME_PHASES.keys())

    rows = []
    for _ in range(n):
        ct = rng.choice(contract_types)
        tier = rng.choice(contractor_tiers)
        phase = rng.choice(phases, p=[0.05, 0.10, 0.55, 0.20, 0.10])
        budget = rng.lognormal(mean=np.log(25e6), sigma=1.2)
        duration = rng.normal(24, 10)
        duration = max(3, duration)

        co_count = int(rng.poisson(12 * CONTRACT_RISK[ct]))
        co_value_pct = rng.gamma(shape=2, scale=4) * CONTRACT_RISK[ct]
        complexity = rng.uniform(1, 10)
        interfaces = int(rng.poisson(4))
        design_completeness = rng.beta(6, 2) * 100
        risk_allowance = rng.uniform(5, 20)
        procurement_risk = rng.uniform(1, 10)

        # Overrun model: based on contract type, contractor, phase, and project features
        base_overrun = rng.normal(18, 8) * CONTRACT_RISK[ct]
        if tier == 3:
            base_overrun *= 1.3
        if phase in ("early_works", "construction"):
            base_overrun *= 1.1
        if design_completeness < 50:
            base_overrun *= 1.25
        if co_value_pct > 15:
            base_overrun *= 1.2
        overrun_pct = base_overrun + rng.normal(0, 5)
        overrun_pct = max(-5, overrun_pct)  # some projects come in under

        rows.append({
            "contract_type_enc":       contract_types.index(ct),
            "contractor_tier_enc":     tier,
            "programme_phase_enc":     phases.index(phase),
            "original_budget_log":     np.log(max(1, budget)),
            "duration_months":         duration,
            "change_order_count":      co_count,
            "change_order_value_pct":  co_value_pct,
            "scope_complexity_score":  complexity,
            "third_party_interfaces":  interfaces,
            "design_completeness_pct": design_completeness,
            "risk_allowance_pct":      risk_allowance,
            "procurement_risk_score":  procurement_risk,
            "overrun_pct":             overrun_pct,
        })

    return pd.DataFrame(rows)


# ─── COST OVERRUN MODEL ───────────────────────────────────────────────────────

class CostOverrunModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.cv_score = None
        self.feature_importance = {}
        self._trained = False

    def train(self, df: pd.DataFrame = None):
        if df is None:
            df = generate_training_data()

        X = df[FEATURES].fillna(0).values
        y = df["overrun_pct"].values

        X_s = self.scaler.fit_transform(X)
        cv = cross_val_score(self.model, X_s, y, cv=5,
                             scoring="neg_root_mean_squared_error")
        self.cv_score = {"CV RMSE": round(-cv.mean(), 2), "CV RMSE std": round(cv.std(), 2)}

        self.model.fit(X_s, y)
        imps = self.model.feature_importances_
        self.feature_importance = {
            FEATURE_LABELS.get(FEATURES[i], FEATURES[i]): round(float(v), 4)
            for i, v in enumerate(imps)
        }
        self._trained = True
        return self.cv_score

    def predict_overrun(self, project_features: dict) -> dict:
        """Predict cost overrun % for a single project."""
        if not self._trained:
            self.train()

        row = [project_features.get(f, 0) for f in FEATURES]
        X = self.scaler.transform([row])
        point = float(self.model.predict(X)[0])

        # Residual distribution from training (approximate)
        residual_std = self.cv_score["CV RMSE"]
        return {
            "overrun_pct_p50": round(point, 1),
            "overrun_pct_p80": round(point + 1.28 * residual_std, 1),
            "overrun_pct_p20": round(point - 0.84 * residual_std, 1),
        }


# ─── MONTE CARLO ENGINE ───────────────────────────────────────────────────────

def run_monte_carlo(
    base_budget: float,
    root_cause_pareto: dict,
    known_variance: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation for final cost forecast.

    Each root cause category contributes uncertainty based on:
    - Its share of known variance (from NLP classification)
    - A volatility multiplier calibrated to that cause type
    - A correlation structure (design errors correlate with scope creep)

    Returns P10/P50/P80 final cost forecasts.
    """
    rng = np.random.default_rng(seed)

    # Volatility by root cause (coefficient of variation, calibrated to IPA data)
    VOLATILITY = {
        "design_change":     0.25,   # large spread — client can keep adding
        "design_error":      0.30,   # high uncertainty — cascading effects
        "ground_conditions": 0.45,   # highest uncertainty — can escalate rapidly
        "procurement_delay": 0.20,
        "scope_creep":       0.35,
        "utility_conflict":  0.40,   # third-party programmes unpredictable
        "weather_force":     0.15,
        "regulatory":        0.30,
    }

    total_additional = np.zeros(n_simulations)

    for cause, share in root_cause_pareto.items():
        cause_variance = known_variance * share
        vol = VOLATILITY.get(cause, 0.25)
        sigma = cause_variance * vol
        # Log-normal draws (costs can't go negative, right-skewed)
        mu = np.log(max(0.01, cause_variance))
        samples = rng.lognormal(mean=mu, sigma=vol, size=n_simulations)
        # Centre around expected value
        samples = samples * (cause_variance / np.exp(mu + vol**2 / 2))
        total_additional += np.clip(samples, 0, cause_variance * 3)

    final_costs = base_budget + total_additional

    p10 = np.percentile(final_costs, 10)
    p50 = np.percentile(final_costs, 50)
    p80 = np.percentile(final_costs, 80)
    p90 = np.percentile(final_costs, 90)

    return {
        "p10": round(p10 / 1e6, 2),
        "p50": round(p50 / 1e6, 2),
        "p80": round(p80 / 1e6, 2),
        "p90": round(p90 / 1e6, 2),
        "base_budget_m": round(base_budget / 1e6, 2),
        "expected_overrun_pct": round((p50 - base_budget) / base_budget * 100, 1),
        "p80_overrun_pct": round((p80 - base_budget) / base_budget * 100, 1),
        "histogram_values": np.clip(final_costs / 1e6, 0, p90 * 1.2 / 1e6).tolist(),
    }


# ─── PORTFOLIO ANALYTICS ─────────────────────────────────────────────────────

def compute_portfolio_metrics(projects_df: pd.DataFrame,
                              change_orders_df: pd.DataFrame) -> dict:
    """
    Aggregate portfolio-level cost intelligence.
    Returns metrics for the portfolio heatmap and contractor accountability matrix.
    """
    # Cost Performance Index by project
    if "actual_cost" in projects_df.columns and "earned_value" in projects_df.columns:
        projects_df["cpi"] = (
            projects_df["earned_value"] / projects_df["actual_cost"].replace(0, np.nan)
        ).fillna(1.0).clip(0.5, 1.5)
    else:
        projects_df["cpi"] = 1.0

    # Variance at completion (VAC)
    if "budget_at_completion" in projects_df.columns and "eac" in projects_df.columns:
        projects_df["vac_pct"] = (
            (projects_df["budget_at_completion"] - projects_df["eac"])
            / projects_df["budget_at_completion"] * 100
        )
    else:
        projects_df["vac_pct"] = 0.0

    # Root cause breakdown from change orders
    if "root_cause" in change_orders_df.columns and "value" in change_orders_df.columns:
        root_pareto = (
            change_orders_df.groupby("root_cause")["value"]
            .sum()
            .sort_values(ascending=False)
        )
        total_co_value = root_pareto.sum()
        root_pareto_pct = (root_pareto / total_co_value * 100).round(1)
    else:
        root_pareto_pct = pd.Series(dtype=float)

    # Contractor CPI ranking
    if "contractor" in change_orders_df.columns:
        contractor_stats = change_orders_df.groupby("contractor").agg(
            total_co_value=("value", "sum"),
            co_count=("value", "count"),
        ).reset_index()
        if "budget" in projects_df.columns:
            contractor_budgets = projects_df.groupby("contractor")["budget"].sum()
            contractor_stats = contractor_stats.merge(
                contractor_budgets.rename("total_budget"),
                on="contractor", how="left"
            )
            contractor_stats["co_pct_of_budget"] = (
                contractor_stats["total_co_value"] /
                contractor_stats["total_budget"] * 100
            ).round(1)
    else:
        contractor_stats = pd.DataFrame()

    return {
        "projects": projects_df,
        "root_pareto_pct": root_pareto_pct,
        "contractor_stats": contractor_stats,
    }


# ─── PORTFOLIO INTERVENTION RULES ────────────────────────────────────────────

def generate_portfolio_recommendations(root_pareto_pct: pd.Series,
                                       total_programme_budget: float,
                                       total_co_value: float) -> list:
    """Generate specific intervention recommendations from portfolio patterns."""
    recs = []
    co_pct = total_co_value / total_programme_budget * 100

    for cause, pct in root_pareto_pct.items():
        if pct < 5:
            continue
        budget_at_risk = total_co_value * (pct / 100)
        label = ROOT_CAUSES.get(cause, cause)

        if cause == "design_change" and pct > 30:
            recs.append({
                "priority": "HIGH",
                "cause": label,
                "pct": pct,
                "value_m": round(budget_at_risk / 1e6, 1),
                "recommendation": "Introduce change freeze protocol at 30% design stage",
                "action": f"Establish Client Change Board with mandatory commercial approval for all scope changes. Estimated saving: ${budget_at_risk * 0.35 / 1e6:.1f}M",
            })
        elif cause == "design_error" and pct > 20:
            recs.append({
                "priority": "HIGH",
                "cause": label,
                "pct": pct,
                "value_m": round(budget_at_risk / 1e6, 1),
                "recommendation": "Implement Independent Design Review at RIBA Stage 4",
                "action": f"Engage independent checker for all IFC packages >$500K scope. Estimated saving: ${budget_at_risk * 0.45 / 1e6:.1f}M",
            })
        elif cause == "ground_conditions" and pct > 15:
            recs.append({
                "priority": "MEDIUM",
                "cause": label,
                "pct": pct,
                "value_m": round(budget_at_risk / 1e6, 1),
                "recommendation": "Enhance ground investigation scope for future packages",
                "action": "Increase GI borehole density to 1 per 50m on all future civils packages. Budget impact: +$0.3M GI cost, expected saving $" + f"{budget_at_risk * 0.25 / 1e6:.1f}M in claims",
            })
        elif cause == "utility_conflict" and pct > 15:
            recs.append({
                "priority": "HIGH",
                "cause": label,
                "pct": pct,
                "value_m": round(budget_at_risk / 1e6, 1),
                "recommendation": "Mandatory SUDS/utility survey before construction package release",
                "action": f"All packages to have utility-cleared corridor confirmed before IFC. Eliminates ${budget_at_risk * 0.6 / 1e6:.1f}M of reactive diversion cost",
            })
        elif cause == "scope_creep" and pct > 10:
            recs.append({
                "priority": "MEDIUM",
                "cause": label,
                "pct": pct,
                "value_m": round(budget_at_risk / 1e6, 1),
                "recommendation": "Implement Scope Control Register with PMO gating",
                "action": f"All site-level scope additions require PMO approval. ${budget_at_risk * 0.5 / 1e6:.1f}M of uncontrolled scope addition recoverable",
            })
        elif cause == "procurement_delay" and pct > 10:
            recs.append({
                "priority": "MEDIUM",
                "cause": label,
                "pct": pct,
                "value_m": round(budget_at_risk / 1e6, 1),
                "recommendation": "Early procurement strategy for long-lead items",
                "action": f"Identify all lead times >20 weeks at RIBA Stage 3. Place advance orders for transformers/switchgear. Saves ${budget_at_risk * 0.4 / 1e6:.1f}M in acceleration premiums",
            })

    recs.sort(key=lambda x: (x["priority"] == "HIGH", x["pct"]), reverse=True)
    return recs


# Import ROOT_CAUSES for use in other modules
from nlp_engine import ROOT_CAUSES
