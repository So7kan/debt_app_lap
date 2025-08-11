# debt_lab.py — Debt Lab with Quick Wizard, Budget Modes, Calendar Timeline, and Back-Solver
# Run:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run debt_lab.py
#
# Features:
# - Input debts via Manual table, CSV/Excel upload, or Quick Wizard
# - Strategies: Avalanche, Snowball, Custom priority
# - Budget modes:
#     A) Extra + optional Cap (budget = sum(mins) + extra; then apply cap if any)
#     B) Total Payment Budget (single number, includes minimums)
# - Date-aware timeline:
#     - Select Start Date
#     - Back-solve the minimal total budget to finish by a Target Date
#     - Forecasted payoff date shown after simulation
# - Full amortization schedule, charts, CSV/JSON export

from __future__ import annotations

import io
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from math import isfinite

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.set_page_config(page_title="Debt Lab", page_icon="💸", layout="wide")


# ---------- Date helpers (no external deps) ----------

def _last_day_of_month(y: int, m: int) -> int:
    if m == 12:
        return 31
    nxt = date(y if m < 12 else y + 1, m + 1 if m < 12 else 1, 1)
    return (nxt - timedelta(days=1)).day

def add_months(d: date, n: int) -> date:
    ym = (d.year * 12 + (d.month - 1)) + n
    y = ym // 12
    m = ym % 12 + 1
    day = min(d.day, _last_day_of_month(y, m))
    return date(y, m, day)

def period_dates(start: date, periods: int, cadence: str) -> List[date]:
    out: List[date] = []
    if periods <= 0:
        return out
    if cadence == "Monthly":
        for k in range(periods):
            out.append(add_months(start, k))
    elif cadence == "Bi-weekly":
        for k in range(periods):
            out.append(start + timedelta(days=14*k))
    else:
        raise ValueError("Unknown cadence")
    return out

def periods_until(start: date, target: date, cadence: str) -> int:
    """Count how many payment periods from start until <= target (inclusive of start if equal)."""
    if target < start:
        return 0
    cnt = 0
    d = start
    if cadence == "Monthly":
        while d <= target:
            cnt += 1
            d = add_months(d, 1)
    elif cadence == "Bi-weekly":
        while d <= target:
            cnt += 1
            d = d + timedelta(days=14)
    else:
        raise ValueError("Unknown cadence")
    return cnt


# ---------- Domain Model ----------

@dataclass
class Debt:
    name: str
    balance: float
    apr_percent: float
    min_payment: float
    due_day: int = 15  # cosmetic

    def monthly_rate(self) -> float:
        return float(self.apr_percent) / 100.0 / 12.0

    def biweekly_rate(self) -> float:
        annual_rate = float(self.apr_percent) / 100.0
        return (1.0 + annual_rate) ** (1.0 / 26.0) - 1.0


@dataclass
class Plan:
    cadence: str
    strategy: str
    # Budget mode: "extra_cap" or "total_budget"
    budget_mode: str
    # Used when budget_mode == "extra_cap"
    extra_payment: float
    payment_cap_total: float
    # Used when budget_mode == "total_budget"
    total_budget_per_period: float
    # Other settings
    custom_priority: List[str]
    start_cash: float
    stop_when_total_paid: bool


# ---------- Utilities ----------

def _round2(x):
    arr = np.round(np.asarray(x, dtype=float) + 1e-12, 2)
    return float(arr) if np.ndim(arr) == 0 else arr

def _order_debts(debts: List[Debt], strategy: str, custom_order: List[str]) -> List[int]:
    if strategy == "Avalanche":
        return sorted(range(len(debts)), key=lambda i: (-debts[i].apr_percent, debts[i].balance))
    elif strategy == "Snowball":
        return sorted(range(len(debts)), key=lambda i: (debts[i].balance, -debts[i].apr_percent))
    else:
        name_to_idx = {d.name: i for i, d in enumerate(debts)}
        pref = [name_to_idx[n] for n in custom_order if n in name_to_idx]
        remaining = [i for i in range(len(debts)) if i not in pref]
        remaining = sorted(remaining, key=lambda i: (-debts[i].apr_percent, debts[i].balance))
        return pref + remaining

def _all_zero(balances: np.ndarray) -> bool:
    return bool(np.all(balances <= 0.005))

def _minsum(debts: List[Debt], alive: np.ndarray) -> float:
    return float(sum(debts[i].min_payment for i in range(len(debts)) if alive[i]))


# ---------- Simulation Engine ----------

def simulate(debts: List[Debt], plan: Plan, max_periods: int = 5000) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
        schedule_df: rows per period per debt with interest, payment, end balance
        meta: summary dict
    """
    n = len(debts)
    if n == 0:
        return pd.DataFrame(), {"error": "No debts"}

    cadence = plan.cadence
    if cadence not in ("Monthly", "Bi-weekly"):
        raise ValueError("Unsupported cadence")

    bal = np.array([float(d.balance) for d in debts], dtype=float)
    alive = bal > 0.005

    # Rates per period
    if cadence == "Monthly":
        r = np.array([d.monthly_rate() for d in debts], dtype=float)
    else:
        r = np.array([d.biweekly_rate() for d in debts], dtype=float)

    # Apply initial lump-sum to current priority list
    order = _order_debts(debts, plan.strategy, plan.custom_priority)
    cash = max(0.0, float(plan.start_cash))
    for i in order:
        if cash <= 0:
            break
        use = min(cash, bal[i])
        bal[i] -= use
        cash -= use
    alive = bal > 0.005

    rows = []
    periods = 0
    cum_interest = 0.0

    for _ in range(max_periods):
        if _all_zero(bal):
            break
        periods += 1

        # 1) Accrue interest
        interest = _round2(bal * r)
        interest = np.where(bal > 0.005, interest, 0.0)
        bal = bal + interest
        cum_interest += float(interest.sum())

        # 2) Budget this period
        alive = bal > 0.005
        min_sum = _minsum(debts, alive)

        if plan.budget_mode == "total_budget":
            budget = float(plan.total_budget_per_period)
            if budget + 1e-9 < min_sum:
                return pd.DataFrame(rows), {
                    "error": f"Total budget (${budget:,.2f}) is less than this period's sum of minimums (${min_sum:,.2f}). "
                             f"Increase your total budget."
                }
        else:
            budget = min_sum + float(plan.extra_payment)
            if plan.payment_cap_total > 0:
                budget = min(budget, float(plan.payment_cap_total))

        if plan.stop_when_total_paid and _all_zero(bal):
            budget = 0.0

        # 3) Pay minimums
        n_debts = len(debts)
        pay = np.zeros(n_debts)
        for i in range(n_debts):
            if not alive[i]:
                continue
            pay[i] = min(debts[i].min_payment, bal[i])
        remaining = max(0.0, budget - float(pay.sum()))

        # 4) Allocate remainder by strategy order
        order = _order_debts(debts, plan.strategy, plan.custom_priority)
        for i in order:
            if remaining <= 0:
                break
            if not alive[i]:
                continue
            need = max(0.0, bal[i] - pay[i])
            use = min(remaining, need)
            pay[i] += use
            remaining -= use

        # 5) Apply payments
        bal = bal - pay
        bal = np.where(bal < 0.005, 0.0, bal)

        # 6) Record
        for i in range(n_debts):
            rows.append({
                "period_index": periods,
                "cadence": cadence,
                "debt": debts[i].name,
                "apr_percent": float(debts[i].apr_percent),
                "period_interest": _round2(interest[i]),
                "period_payment": _round2(pay[i]),
                "end_balance": _round2(bal[i]),
            })

        # Safety check: ensure progress
        if periods > 10000:
            return pd.DataFrame(rows), {"error": "Simulation exceeded safe period limit."}

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return schedule, {"error": "No schedule produced (check inputs)."}

    by_debt = schedule.groupby("debt", as_index=False).agg(
        total_interest=("period_interest", "sum"),
        total_paid=("period_payment", "sum"),
        last_balance=("end_balance", "last"),
        periods=("period_index", "max"),
    )
    total_interest = float(by_debt["total_interest"].sum())
    total_paid = float(by_debt["total_paid"].sum())
    max_periods_used = int(schedule["period_index"].max() if not schedule.empty else 0)

    months_equiv = max_periods_used if cadence == "Monthly" else max_periods_used * 12.0 / 26.0
    years_equiv = months_equiv / 12.0

    meta = {
        "periods_used": max_periods_used,
        "cadence": cadence,
        "months_equiv": months_equiv,
        "years_equiv": years_equiv,
        "total_interest": _round2(total_interest),
        "total_paid": _round2(total_paid),
    }
    return schedule, meta


# ---------- Back-solver ----------

def backsolve_budget_for_deadline(
    debts: List[Debt],
    base_plan: Plan,
    start_date: date,
    target_date: date,
    max_iterations: int = 40,
) -> Dict[str, Optional[float]]:
    """
    Find the minimal total budget per period that pays everything by target_date.
    Returns dict with keys: ok (bool), budget (float or None), reason (str)
    """
    if target_date <= start_date:
        return {"ok": False, "budget": None, "reason": "Target date must be after start date."}

    target_periods = periods_until(start_date, target_date, base_plan.cadence)
    if target_periods <= 0:
        return {"ok": False, "budget": None, "reason": "No periods available between dates."}

    # Lower bound: initial sum of minimums (first period). Must be feasible.
    # We'll test feasibility as we go because future min_sum can only go down.
    # Upper bound: find a budget that surely finishes within target periods by doubling.
    low = 0.0  # we'll raise low to the first-period min_sum on the first probe
    # Probe to get first-period min sum:
    tmp_plan = Plan(
        cadence=base_plan.cadence,
        strategy=base_plan.strategy,
        budget_mode="total_budget",
        extra_payment=0.0,
        payment_cap_total=0.0,
        total_budget_per_period=0.0,  # placeholder
        custom_priority=base_plan.custom_priority,
        start_cash=base_plan.start_cash,
        stop_when_total_paid=base_plan.stop_when_total_paid,
    )

    # Compute first-period minimum sum
    # Hack: run one tiny step by calling simulate with huge budget but capture first min_sum?
    # Cleaner: compute directly:
    first_min_sum = float(sum(d.min_payment for d in debts if d.balance > 0.005))
    if first_min_sum <= 0.0:
        return {"ok": True, "budget": 0.0, "reason": "No active minimums — debts might already be zero."}

    low = first_min_sum
    high = max(low, 50_000.0)
    # Increase high until payoff periods <= target_periods (or until obviously huge)
    for _ in range(20):
        tmp_plan.total_budget_per_period = high
        sched, meta = simulate(debts, tmp_plan)
        if "error" not in meta and meta.get("periods_used", 10**9) <= target_periods:
            break
        high *= 2.0
        if high > 10_000_000:
            return {"ok": False, "budget": None, "reason": "Could not bracket solution (inputs may be inconsistent)."}

    # Binary search
    best = None
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        tmp_plan.total_budget_per_period = mid
        sched, meta = simulate(debts, tmp_plan)
        if "error" in meta:
            # Budget < min_sum somewhere — infeasible ⇒ raise low
            low = mid
            continue

        used = meta.get("periods_used", 10**9)
        if used <= target_periods:
            best = mid
            high = mid
        else:
            low = mid

        if abs(high - low) < 0.01:
            break

    if best is None:
        return {"ok": False, "budget": None, "reason": "No feasible budget found for that deadline."}
    return {"ok": True, "budget": round(best + 1e-9, 2), "reason": ""}


# ---------- Example + Upload ----------

def example_df():
    return pd.DataFrame([
        {"name": "AMEX", "balance": 9764, "apr_percent": 9.0, "min_payment": 120, "due_day": 15},
        {"name": "MBNA", "balance": 8000, "apr_percent": 0.0, "min_payment": 313.85, "due_day": 1},
        {"name": "Triangle", "balance": 8653.87, "apr_percent": 19.99, "min_payment": 90, "due_day": 10},
        {"name": "RBC LOC", "balance": 14782, "apr_percent": 12.5, "min_payment": 180, "due_day": 20},
        {"name": "BMO LOC", "balance": 7366, "apr_percent": 13.99, "min_payment": 120, "due_day": 12},
    ])

def df_from_upload(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file)
    except Exception:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return pd.DataFrame()
    needed = {"name", "balance", "apr_percent", "min_payment"}
    lower_cols = {c.lower(): c for c in df.columns}
    missing = needed - set(lower_cols)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        return pd.DataFrame()
    out = pd.DataFrame({
        "name": df[lower_cols["name"]],
        "balance": pd.to_numeric(df[lower_cols["balance"]], errors="coerce").fillna(0.0),
        "apr_percent": pd.to_numeric(df[lower_cols["apr_percent"]], errors="coerce").fillna(0.0),
        "min_payment": pd.to_numeric(df[lower_cols["min_payment"]], errors="coerce").fillna(0.0),
        "due_day": pd.to_numeric(df[lower_cols["due_day"]] if "due_day" in lower_cols else 15, errors="coerce").fillna(15).astype(int),
    })
    return out


# ---------- App UI ----------

st.title("💸 Debt Lab — Avalanche / Snowball Simulator")
st.caption("Choose Manual, CSV Upload, or Quick Wizard. Pick a budget mode, simulate, view calendar payoff, export results.")

with st.sidebar:
    st.subheader("Input Mode")
    mode = st.radio("How will you provide debts?", ["Manual", "Upload CSV", "Quick Wizard"], horizontal=False)

    if "wizard_df" not in st.session_state:
        st.session_state.wizard_df = pd.DataFrame()

    if mode == "Upload CSV":
        up = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
        df = df_from_upload(up)
        if df.empty:
            st.info("No valid upload detected — falling back to example.")
            df = example_df()

    elif mode == "Manual":
        st.caption("Start from example and edit inline. (Use the table to modify values.)")
        df = example_df()

    else:  # Quick Wizard
        st.caption("Generate a starter table fast, then fine-tune inline.")
        count = st.number_input("Number of debts", min_value=1, max_value=50, value=5, step=1)
        prefix = st.text_input("Name prefix", value="Debt")
        default_bal = st.number_input("Default balance for new rows", min_value=0.0, value=5000.0, step=100.0)
        default_apr = st.number_input("Default APR % for new rows", min_value=0.0, value=14.99, step=0.25)
        default_min = st.number_input("Default minimum payment", min_value=0.0, value=100.0, step=10.0)
        default_due = st.number_input("Default due day (1–28)", min_value=1, max_value=28, value=15, step=1)

        colg1, colg2 = st.columns([1,1])
        with colg1:
            gen = st.button("Generate / Reset Table", use_container_width=True)
        with colg2:
            keep = st.button("Keep Current Table", use_container_width=True)

        if gen:
            st.session_state.wizard_df = pd.DataFrame({
                "name": [f"{prefix} {i+1}" for i in range(int(count))],
                "balance": [default_bal]*int(count),
                "apr_percent": [default_apr]*int(count),
                "min_payment": [default_min]*int(count),
                "due_day": [default_due]*int(count),
            })

        if st.session_state.wizard_df.empty:
            st.info("Click **Generate / Reset Table** to create rows.")
            df = pd.DataFrame(columns=["name","balance","apr_percent","min_payment","due_day"])
        else:
            df = st.session_state.wizard_df.copy()

    st.divider()
    st.subheader("Plan")

    cadence = st.selectbox("Payment cadence", ["Monthly", "Bi-weekly"])
    strategy = st.selectbox("Strategy", ["Avalanche", "Snowball", "Custom"])

    budget_mode_label = st.radio(
        "Budget mode",
        options=["Extra + optional Cap", "Total Payment Budget"],
        index=0,
        help="How do you want to set your payment budget per period?",
    )
    if budget_mode_label == "Extra + optional Cap":
        extra_payment = st.number_input(
            "Extra payment per period (on top of minimums)",
            min_value=0.0, value=500.0, step=50.0
        )
        payment_cap = st.number_input(
            "Total payment cap per period (0 = unlimited)",
            min_value=0.0, value=0.0, step=50.0
        )
        total_budget = 0.0
    else:
        total_budget = st.number_input(
            "Total payment budget per period (includes minimums)",
            min_value=0.0, value=1500.0, step=50.0,
            help="Example: if minimums sum to $1,000 and you set $1,500, the extra $500 goes to the priority debt."
        )
        extra_payment = 0.0
        payment_cap = 0.0

    start_cash = st.number_input("Initial lump-sum to deploy immediately", min_value=0.0, value=0.0, step=100.0)
    stop_when_total_paid = st.checkbox("Stop spending once debts finish (don’t roll freed budget)", value=False)

    # Calendar controls
    st.divider()
    st.subheader("Calendar")
    start_dt = st.date_input("Start date", value=date.today())
    with st.expander("Back-solve a budget for a target payoff date"):
        target_dt = st.date_input("Target payoff date", value=add_months(date.today(), 24))
        if st.button("Back-solve required total budget", use_container_width=True):
            # Build temporary base plan for back-solve (strategy + cadence + settings)
            tmp_plan = Plan(
                cadence=cadence,
                strategy=strategy,
                budget_mode="total_budget",
                extra_payment=0.0,
                payment_cap_total=0.0,
                total_budget_per_period=0.0,
                custom_priority=[],
                start_cash=start_cash,
                stop_when_total_paid=stop_when_total_paid,
            )
            # Build debts from current df for the solver
            tmp_debts: List[Debt] = []
            for _, r in df.iterrows():
                name = str(r.get("name","")).strip()
                if not name: 
                    continue
                try:
                    tmp_debts.append(Debt(
                        name=name,
                        balance=float(r.get("balance", 0.0)),
                        apr_percent=float(r.get("apr_percent", 0.0)),
                        min_payment=float(r.get("min_payment", 0.0)),
                        due_day=int(r.get("due_day", 15)),
                    ))
                except Exception:
                    continue

            if not tmp_debts:
                st.error("Add at least one debt before back-solving.")
            else:
                res = backsolve_budget_for_deadline(tmp_debts, tmp_plan, start_dt, target_dt)
                if not res["ok"]:
                    st.error(res["reason"])
                else:
                    st.success(f"Minimum total budget per period to finish by {target_dt}: ${res['budget']:.2f}")
                    st.session_state["suggested_budget"] = res["budget"]

        # Option to adopt suggested budget
        if "suggested_budget" in st.session_state and budget_mode_label == "Total Payment Budget":
            if st.button(f"Use suggested budget (${st.session_state['suggested_budget']:.2f})", use_container_width=True):
                total_budget = float(st.session_state["suggested_budget"])
                st.success(f"Applied total budget: ${total_budget:.2f}")

# Editable debts table
st.subheader("Debts")
editor_key = f"debts_editor_{mode}"
ed = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "name": st.column_config.TextColumn("Name"),
        "balance": st.column_config.NumberColumn("Balance", format="%.2f"),
        "apr_percent": st.column_config.NumberColumn("APR %", format="%.2f"),
        "min_payment": st.column_config.NumberColumn("Min Payment", format="%.2f"),
        "due_day": st.column_config.NumberColumn("Due Day", min_value=1, max_value=28),
    },
    hide_index=True,
    key=editor_key
)

if mode == "Quick Wizard":
    st.session_state.wizard_df = ed.copy()

# Build Debt objects
debts: List[Debt] = []
for _, r in ed.iterrows():
    name = str(r.get("name","")).strip()
    if not name:
        continue
    try:
        debts.append(Debt(
            name=name,
            balance=float(r.get("balance", 0.0)),
            apr_percent=float(r.get("apr_percent", 0.0)),
            min_payment=float(r.get("min_payment", 0.0)),
            due_day=int(r.get("due_day", 15)),
        ))
    except Exception:
        continue

# Custom ordering UI
custom_order: List[str] = []
strategy = strategy  # from sidebar
if strategy == "Custom" and debts:
    st.subheader("Custom Priority Order")
    names = [d.name for d in debts]
    custom_order = st.multiselect(
        "Choose payoff priority (top = highest priority). Any not selected will follow avalanche.",
        names,
        default=names,
    )

plan = Plan(
    cadence=cadence,
    strategy=strategy,
    budget_mode=("extra_cap" if budget_mode_label == "Extra + optional Cap" else "total_budget"),
    extra_payment=float(extra_payment),
    payment_cap_total=float(payment_cap),
    total_budget_per_period=float(total_budget),
    custom_priority=custom_order,
    start_cash=float(start_cash),
    stop_when_total_paid=bool(stop_when_total_paid),
)

run = st.button("Run Simulation", type="primary", use_container_width=True)

if run:
    if not debts:
        st.error("Add at least one debt.")
        st.stop()

    schedule, meta = simulate(debts, plan)
    if "error" in meta:
        st.error(meta["error"])
        st.stop()

    # Attach calendar dates to schedule
    periods_used = int(meta["periods_used"])
    dates_list = period_dates(start_dt, periods_used, plan.cadence)
    # Map period_index -> date
    period_to_date = {i+1: dates_list[i] for i in range(len(dates_list))}
    schedule["date"] = schedule["period_index"].map(period_to_date)

    payoff_date = dates_list[-1] if dates_list else None

    st.success(
        f"Plan complete in ~{meta['months_equiv']:.1f} months (~{meta['years_equiv']:.2f} years). "
        f"Total interest: ${meta['total_interest']:.2f}. "
        + (f"Forecast payoff date: **{payoff_date}**." if payoff_date else "")
    )

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Cadence", meta["cadence"])
    colB.metric("Periods", meta["periods_used"])
    colC.metric("Months (equiv.)", f"{meta['months_equiv']:.1f}")
    colD.metric("Total Interest", f"${meta['total_interest']:.2f}")

    st.divider()
    st.subheader("Totals by Account")
    by_debt = schedule.groupby("debt", as_index=False).agg(
        periods=("period_index", "max"),
        final_balance=("end_balance", "last"),
        total_paid=("period_payment", "sum"),
        interest_paid=("period_interest", "sum"),
        apr=("apr_percent", "max"),
    )
    st.dataframe(by_debt, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Total Balance Over Time", "Interest vs. Payment", "Per-Account Trajectories"])

    with tab1:
        gb = schedule.groupby("period_index", as_index=False)["end_balance"].sum()
        fig = plt.figure()
        plt.plot(gb["period_index"], gb["end_balance"])
        plt.title("Total Balance Over Time")
        plt.xlabel("Period")
        plt.ylabel("Total End Balance")
        st.pyplot(fig)

    with tab2:
        gi = schedule.groupby("period_index", as_index=False)[["period_interest", "period_payment"]].sum()
        fig2 = plt.figure()
        plt.plot(gi["period_index"], gi["period_interest"], label="Interest This Period")
        plt.plot(gi["period_index"], gi["period_payment"], label="Payment This Period")
        plt.title("Interest vs. Payments")
        plt.xlabel("Period")
        plt.ylabel("Amount")
        plt.legend()
        st.pyplot(fig2)

    with tab3:
        debts_list = sorted(schedule["debt"].unique().tolist())
        pick = st.multiselect("Choose accounts to plot", debts_list, default=debts_list[:min(5, len(debts_list))])
        fig3 = plt.figure()
        for dname in pick:
            sd = schedule[schedule["debt"] == dname]
            plt.plot(sd["period_index"], sd["end_balance"], label=dname)
        plt.title("Per-Account Paydown")
        plt.xlabel("Period")
        plt.ylabel("End Balance")
        if pick:
            plt.legend()
        st.pyplot(fig3)

    st.divider()
    st.subheader("Amortization Schedule")
    st.dataframe(schedule, use_container_width=True, height=380)

    st.subheader("Export")
    csv_bytes = schedule.to_csv(index=False).encode("utf-8")
    st.download_button("Download Schedule CSV", data=csv_bytes, file_name="debt_schedule.csv", mime="text/csv")

    scenario = {
        "debts": [asdict(d) for d in debts],
        "plan": asdict(plan),
        "meta": {**meta, "start_date": str(start_dt), "payoff_date": str(payoff_date) if payoff_date else None},
    }
    json_bytes = json.dumps(scenario, indent=2).encode("utf-8")
    st.download_button("Download Scenario JSON", data=json_bytes, file_name="debt_scenario.json", mime="application/json")

else:
    st.info("Pick an input mode, adjust the table, set your plan + calendar, then click **Run Simulation**.")
