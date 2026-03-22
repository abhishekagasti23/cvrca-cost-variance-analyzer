"""
CVRCA Demo Data Generator
Synthetic US Grid Modernization Programme — IIJA/IRA-funded capital portfolio.
15 projects, 300+ change orders with realistic cost variance patterns.
Anchor: IIJA-funded transmission and distribution upgrade, typical US regulated utility.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# ─── PROGRAMME STRUCTURE ─────────────────────────────────────────────────────

PROJECTS = [
    # (id, name, budget_m, contractor, tier, contract_type, complexity, interfaces, start_offset_days)
    ("P01", "345kV Transmission Line — Northern Corridor",   185.0, "Bechtel Energy",       "tier1", "lump_sum",    8.5, 4,  0),
    ("P02", "Substation Upgrade — Eagle Rock 230/115kV",     62.0,  "AECOM Infrastructure", "tier2", "remeasure",   6.0, 3,  30),
    ("P03", "Distribution Automation — Zone 3 SCADA",        28.5,  "Quanta Services",      "tier2", "target_cost", 7.5, 5,  45),
    ("P04", "Underground Cable Replacement — Urban Core",    47.0,  "MYR Group",            "tier2", "remeasure",   7.0, 6,  20),
    ("P05", "Smart Meter Rollout — 80,000 Residential",      38.0,  "Itron Inc.",            "tier1", "lump_sum",    4.5, 2,  15),
    ("P06", "Bulk Power Storage — 200MWh BESS",              95.0,  "Fluor Corporation",    "tier1", "design_build",9.0, 3,  60),
    ("P07", "Transmission Line Reconductoring — Phase 1",    31.0,  "Primoris Services",    "tier2", "remeasure",   5.5, 2,  90),
    ("P08", "Substation Automation — 12 Sites",              22.0,  "S&C Electric",         "tier2", "target_cost", 6.5, 4, 120),
    ("P09", "High-Voltage DC Link — Phase 1",               240.0,  "Bechtel Energy",       "tier1", "lump_sum",   9.5, 7,  10),
    ("P10", "Wildfire Hardening — High-Risk Zones",          55.0,  "Quanta Services",      "tier2", "remeasure",   6.0, 3,  75),
    ("P11", "Telecommunications Infrastructure Upgrade",     18.0,  "Local Electric LLC",   "tier3", "cost_plus",   5.0, 4, 100),
    ("P12", "Transmission Switching Station — Desert View",  78.0,  "AECOM Infrastructure", "tier2", "lump_sum",    8.0, 5,  50),
    ("P13", "EV Charging Infrastructure — Highway Corridor", 24.0,  "MYR Group",            "tier2", "target_cost", 5.5, 4,  80),
    ("P14", "Grid Resilience Upgrades — Storm Hardening",    44.0,  "Primoris Services",    "tier2", "remeasure",   6.5, 3,  40),
    ("P15", "Microgrid Pilot — Industrial Zone",             31.0,  "S&C Electric",         "tier2", "design_build",7.0, 5,  55),
]

# Change order description templates per root cause
CO_TEMPLATES = {
    "design_change": [
        "Client instruction to upgrade conductor size from 795 ACSR to 1272 ACSR on spans 12-18",
        "Owner directed addition of fiber optic ground wire (OPGW) not in original scope",
        "Employer instruction EI-{n} to include additional switching positions at {loc}",
        "Client requested change to BESS chemistry from LFP to NMC for higher energy density",
        "Owner directed scope addition: integration with ISO real-time pricing system",
        "Client instruction to increase substation capacity from 200MVA to 250MVA",
        "Variation order: additional protection relay upgrade for N-2 redundancy",
        "Client directed change to tower design for increased future capacity",
        "Owner instruction to add communications redundancy not in original specification",
        "Client requested acceleration of {section} to meet FERC compliance deadline",
        "Employer directed addition of seismic restraints to all equipment over 500kg",
        "Client instruction to incorporate AMI 2.0 communications module in meter specification",
    ],
    "design_error": [
        "RFI-{n}: Foundation design inadequate for actual soil bearing capacity at tower {loc}",
        "Drawing conflict: cable tray routing conflicts with structural steelwork at substation bay",
        "Design omission: no provision for surge arrester on cable termination at {loc}",
        "Specification error: incorrect insulation level specified for 230kV equipment",
        "RFI raised: grounding grid design insufficient per IEEE 80 calculations",
        "Drawing discrepancy between civil and electrical layouts at cable vault {loc}",
        "Design deficiency: conductor sag calculations did not account for ACSR creep",
        "Missing detail: cable tray support spacing not specified for seismic zone 4",
        "RFI-{n}: conflicts between mechanical and electrical penetration drawings at {loc}",
        "Incorrect assumed cable pulling tension leads to duct bank redesign",
        "Design coordination error: transformer oil containment sump undersized",
        "Specification ambiguity in fire protection system for battery room",
    ],
    "ground_conditions": [
        "Rock encountered at tower foundation {loc} requiring blasting: GI showed soil",
        "Contaminated soil discovered during excavation at former industrial site Ch {loc}",
        "Unforeseen soft ground at cable route {loc} requiring timber pile crib stabilization",
        "Buried petroleum pipeline discovered during trench excavation at {loc}",
        "Groundwater levels 3ft higher than boring data indicated — dewatering required",
        "Unexpected caliche layer at {loc} requiring jackhammer excavation",
        "Abandoned mine workings discovered below tower foundation location {loc}",
        "Contamination plume from former gas station requires hazmat disposal",
        "Hard rock head significantly shallower than assumed — tower foundation redesign",
        "Unexpected buried concrete foundations from former transmission line at {loc}",
    ],
    "procurement_delay": [
        "345kV transformer delivery delayed 16 weeks by manufacturer — acceleration required",
        "Long-lead SCADA RTU not available within programme: substitute sourced at premium",
        "Copper conductors delayed due to supply chain disruption — commodity cost +34%",
        "Power transformer OEM reports 52-week lead time due to global capacity constraint",
        "Steel lattice tower sections delayed at fabricator: import from Canada at premium",
        "Specialty cable delivery delayed 8 weeks — storage costs and extended site presence",
        "Circuit breaker lead time extended to 60 weeks: alternative GIS source identified",
        "Aluminum conductor price increase of 28% from tender to order date",
        "BESS battery modules delayed at port: container shortage impacting supply chain",
        "Subcontractor insolvency mid-procurement requires re-tender at higher market price",
    ],
    "scope_creep": [
        "Additional cable containment installed for future capacity not in contract scope",
        "Extra civil works added progressively during construction: uncontrolled expansion",
        "SCADA points list expanded by 40% during commissioning without formal change order",
        "Additional testing requirements added at site level beyond specification",
        "Unplanned access road improvements required by local municipality",
        "Additional grounding connections added by field engineer without instruction",
        "Extra security fencing installed beyond contract specification at 3 sites",
        "Software functionality added to SCADA HMI beyond original specification",
        "Additional signage and marking installed beyond required specification",
        "Scope growth in protection relay testing protocol beyond original scope",
    ],
    "utility_conflict": [
        "Uncharted fiber optic cable conflict at tower {loc} requiring telecom company diversion",
        "Gas distribution main in conflict with cable route at {loc}: utility company diversion",
        "Water main conflict discovered at cable vault location {loc}: municipal water dept",
        "Railroad crossing conflict requires UP Railroad design approval — 12 week process",
        "Existing 69kV distribution line in conflict with new 345kV ROW — relocation required",
        "Municipal sewer in conflict with substation cable trench: city utility engagement",
        "Stormwater drain conflict at {loc} requires County coordination and redesign",
        "Telecom duct bank conflict discovered during potholing at {loc}",
        "Buried irrigation system conflict with cable route in agricultural section",
        "Third-party fiber optic network owner refuses to relocate within programme",
    ],
    "weather_force": [
        "Exceptional rainfall in Q3 — 1-in-20 year event: earthworks productivity 35% below plan",
        "Ice storm caused 3-week shutdown of transmission line erection operations",
        "High winds prevented crane operations for 9 working days in November",
        "Flash flooding of access road and cable trenches: pump-out and reinstatement required",
        "Extreme heat wave: concrete pours restricted to night shifts for 3 weeks",
        "Wildfire proximity — site evacuated for 5 days, equipment secured",
        "Winter storm Elliot: site closed for 7 days, frozen ground prevents excavation",
        "Hurricane watch: tower erection suspended, equipment secured for 4 days",
        "Extreme cold snap: soil frozen to 4ft depth precluding foundation excavation",
        "Persistent fog preventing helicopter conductor stringing operations",
    ],
    "regulatory": [
        "FERC order revision requires additional protection relay functionality",
        "State PUC condition requires enhanced vegetation management plan — extra cost",
        "Environmental permit condition requires additional stormwater management",
        "Discharge of CUP condition 12 delayed 8 weeks by county planning",
        "NERC CIP compliance requirement added post-award for cyber security",
        "FAA obstruction lighting requirement added for towers over 200ft AGL",
        "USACE Section 404 permit more restrictive than anticipated — wetlands mitigation",
        "Updated NEC code cycle requires additional grounding provisions",
        "OSHA citation on similar project requires method statement revision",
        "Tribal consultation requirement delays ROW acquisition by 10 weeks",
    ],
}

LOCATIONS = [
    "Sta 1+450", "Sta 4+200", "Sta 7+800", "Sta 12+100", "Bay 3A",
    "Bay 5B", "Tower 14", "Tower 27", "Tower 41", "Vault B-7",
    "Section 2", "Section 5", "Segment A", "Zone 3", "Building C",
]


def generate_demo_data(seed: int = 42):
    """
    Generate a complete synthetic US grid modernization portfolio.
    Returns: (projects_df, change_orders_df, cost_tracker_df)
    """
    rng = np.random.default_rng(seed)
    programme_start = datetime(2022, 10, 1)
    data_date = datetime(2025, 1, 15)

    from nlp_engine import ROOT_CAUSES

    # ── Projects ──────────────────────────────────────────────────────────────
    project_rows = []
    for pid, name, budget_m, contractor, tier, ctype, complexity, interfaces, offset in PROJECTS:
        start = programme_start + timedelta(days=offset)
        duration_months = rng.integers(18, 48)
        end_planned = start + timedelta(days=int(duration_months * 30.5))

        # Overrun factor
        from cost_model import CONTRACT_RISK
        overrun_factor = rng.normal(1.0 + 0.18 * CONTRACT_RISK[ctype], 0.08)
        overrun_factor = max(0.95, overrun_factor)

        budget = budget_m * 1e6
        actual_cost_to_date = budget * rng.uniform(0.35, 0.75)
        earned_value = actual_cost_to_date * rng.normal(0.92, 0.07)
        eac = budget * overrun_factor
        acwp = actual_cost_to_date
        bcwp = earned_value

        pct_complete = min(95, max(5, rng.normal(50, 20)))

        project_rows.append({
            "project_id": pid,
            "project_name": name,
            "contractor": contractor,
            "contractor_tier": tier,
            "contract_type": ctype,
            "sector": "Energy & Utilities",
            "budget": budget,
            "budget_m": budget_m,
            "eac": eac,
            "actual_cost": acwp,
            "earned_value": bcwp,
            "pct_complete": round(pct_complete, 1),
            "start_date": start,
            "planned_end": end_planned,
            "duration_months": int(duration_months),
            "scope_complexity": complexity,
            "third_party_interfaces": interfaces,
            "overrun_pct": round((overrun_factor - 1) * 100, 1),
            "cpi": round(bcwp / max(1, acwp), 3),
            "vac_m": round((budget - eac) / 1e6, 2),
            "status": rng.choice(["Active", "Active", "Active", "Delayed", "On Track"],
                                  p=[0.4, 0.3, 0.1, 0.15, 0.05]),
        })

    projects_df = pd.DataFrame(project_rows)

    # ── Change Orders ─────────────────────────────────────────────────────────
    from nlp_engine import ROOT_CAUSES as RC
    root_cause_keys = list(RC.keys())

    # Frequency weights per cause (calibrated to US infra data)
    cause_weights = np.array([0.22, 0.18, 0.12, 0.13, 0.10, 0.11, 0.07, 0.07])
    cause_weights /= cause_weights.sum()

    co_rows = []
    co_id = 1000

    for _, proj in projects_df.iterrows():
        # Number of COs roughly proportional to budget and contract type
        from cost_model import CONTRACT_RISK as CR
        n_cos = int(rng.poisson(10 * CR[proj["contract_type"]]))
        n_cos = max(3, min(35, n_cos))

        project_co_value = proj["budget"] * rng.uniform(0.08, 0.28)

        for j in range(n_cos):
            cause = rng.choice(root_cause_keys, p=cause_weights)
            templates = CO_TEMPLATES[cause]
            tmpl = templates[int(rng.integers(len(templates)))]
            loc = LOCATIONS[int(rng.integers(len(LOCATIONS)))]
            n_val = int(rng.integers(10, 200))
            description = tmpl.replace("{loc}", loc).replace("{n}", str(n_val)).replace("{section}", f"Section {int(rng.integers(1,6))}")

            # Value: log-normal, scaled to project total CO budget
            value = rng.lognormal(mean=np.log(project_co_value / n_cos), sigma=0.6)
            value = max(5_000, min(value, proj["budget"] * 0.08))

            co_date = proj["start_date"] + timedelta(days=int(rng.integers(30, proj["duration_months"] * 25)))
            co_date = min(co_date, data_date)

            status = rng.choice(
                ["Approved", "Approved", "Approved", "Pending", "Rejected"],
                p=[0.55, 0.15, 0.10, 0.15, 0.05]
            )

            co_rows.append({
                "co_id": f"CO-{co_id:04d}",
                "project_id": proj["project_id"],
                "project_name": proj["project_name"],
                "contractor": proj["contractor"],
                "date": co_date,
                "description": description,
                "root_cause": cause,  # ground truth for demo
                "root_cause_label": RC[cause],
                "value": round(value, 0),
                "value_m": round(value / 1e6, 3),
                "status": status,
                "approved_value": round(value * rng.uniform(0.7, 1.0) if status != "Rejected" else 0, 0),
            })
            co_id += 1

    co_df = pd.DataFrame(co_rows)
    co_df["date"] = pd.to_datetime(co_df["date"])

    # ── Cost Tracker (monthly ACWP / BCWP) ────────────────────────────────────
    tracker_rows = []
    for _, proj in projects_df.iterrows():
        months = proj["duration_months"]
        budget = proj["budget"]
        eac = proj["eac"]
        months_elapsed = max(1, min(months, int((data_date - proj["start_date"]).days / 30.5)))

        for m in range(1, months_elapsed + 1):
            # S-curve with overrun
            t = m / months
            s = t**2 * (3 - 2 * t)  # S-curve
            planned_cumulative = budget * s
            # Actual: track overrun gradually materialising
            overrun_at_m = 1 + (proj["overrun_pct"] / 100) * min(1, t * 1.5)
            actual_cumulative = planned_cumulative * overrun_at_m * rng.uniform(0.97, 1.03)
            earned_cumulative = planned_cumulative * rng.normal(0.93, 0.04)

            tracker_rows.append({
                "project_id": proj["project_id"],
                "month": m,
                "date": proj["start_date"] + timedelta(days=int(m * 30.5)),
                "planned_cumulative": round(planned_cumulative, 0),
                "actual_cumulative": round(actual_cumulative, 0),
                "earned_cumulative": round(max(0, earned_cumulative), 0),
                "cpi_monthly": round(earned_cumulative / max(1, actual_cumulative), 3),
            })

    tracker_df = pd.DataFrame(tracker_rows)

    return projects_df, co_df, tracker_df


def to_excel(projects_df, co_df):
    """Export demo data to Excel for upload testing."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Projects sheet
        p_export = projects_df[[
            "project_id", "project_name", "contractor", "contract_type",
            "budget_m", "eac", "pct_complete", "overrun_pct", "status",
        ]].copy()
        p_export.columns = [
            "Project ID", "Project Name", "Contractor", "Contract Type",
            "Budget ($M)", "EAC ($M)", "% Complete", "Overrun %", "Status",
        ]
        p_export["EAC ($M)"] = (p_export["EAC ($M)"] / 1e6).round(2)
        p_export.to_excel(writer, sheet_name="Projects", index=False)

        # Change orders sheet — description only (no root_cause = what the NLP classifies)
        co_export = co_df[[
            "co_id", "project_id", "date", "description", "value", "status"
        ]].copy()
        co_export.columns = [
            "CO ID", "Project ID", "Date", "Description", "Value ($)", "Status"
        ]
        co_export.to_excel(writer, sheet_name="Change Orders", index=False)
    return buf.getvalue()
