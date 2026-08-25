"""
Toy simulation for intelligent allocation of delinquent clients to collection agents.

Model:
- 20 clients, 4 agents, 4 one-hour slots: 09, 10, 11, 12.
- Each client can be called at most once per day.
- Each agent can call exactly one client per hour.
- Response probability is client x hour and is fixed across Monte-Carlo runs.
- Agent-client effectiveness is agent x client, redrawn once per simulated day.
  Its matrix mean is forced to exactly AGENT_EFF_MEAN in every simulation.
- All debts are equal, so maximizing expected collected amount is equivalent
  to maximizing expected number of successful collections.
- Success probability for a call:
      q(client, agent, hour) = p_response(client, hour) * agent_effectiveness(agent, client)

Policies:
1. Random
2. Carousel: top current response probability, sequential assignment to agents
3. Hourly greedy: current-hour max-weight matching using q
4. Full-horizon optimum: max-weight matching over all 16 day slots
5. Rolling horizon: re-solves remaining horizon each hour

The experiment runs N_SIMULATIONS full-day Monte-Carlo simulations.
Therefore each policy has N_SIMULATIONS observations for each hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


# ============================================================
# Configuration
# ============================================================

SEED = 42

N_CLIENTS = 20
N_AGENTS = 4
HOURS = np.array([9, 10, 11, 12], dtype=int)
N_HOURS = len(HOURS)

N_SIMULATIONS = 1_000
DEBT_PER_CLIENT = 100_000.0

# Small response probabilities.
MIN_RESPONSE_P = 0.02
MAX_RESPONSE_P = 0.30

# Mean is exactly this value in every simulated agent-client matrix.
AGENT_EFF_MEAN = 0.60
AGENT_EFF_HALF_RANGE = 0.15

OUTPUT_DIR = Path("simulation_output")

POLICY_ORDER = [
    "Random",
    "Carousel",
    "Hourly greedy",
    "Full-horizon optimum",
    "Rolling horizon",
]


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class Assignment:
    hour_idx: int
    agent: int
    client: int


# ============================================================
# Synthetic data
# ============================================================

def make_response_probabilities(rng: np.random.Generator) -> np.ndarray:
    """
    Create fixed client x hour response probabilities.

    We intentionally give clients different intraday shapes, so some customers
    are more valuable later and should be held back by a horizon-aware policy.
    """
    base = rng.uniform(0.04, 0.15, size=N_CLIENTS)

    # Four simple intraday patterns.
    patterns = np.array([
        [0.85, 0.95, 1.15, 1.45],  # rising through the day
        [1.40, 1.15, 0.95, 0.75],  # falling through the day
        [0.90, 1.40, 1.10, 0.90],  # peak at 10
        [0.85, 1.00, 1.45, 1.05],  # peak at 11
    ])

    p = np.empty((N_CLIENTS, N_HOURS), dtype=float)

    for client in range(N_CLIENTS):
        intraday = patterns[client % len(patterns)]
        small_noise = rng.normal(loc=1.0, scale=0.06, size=N_HOURS)
        p[client] = base[client] * intraday * small_noise

    p = np.clip(p, MIN_RESPONSE_P, MAX_RESPONSE_P)

    # Explicit example of a client that is much more attractive later.
    p[0] = np.array([0.04, 0.06, 0.15, 0.28])

    return p


def make_agent_effectiveness(rng: np.random.Generator) -> np.ndarray:
    """
    Agent x client matrix, redrawn once per simulated day.

    The matrix mean is EXACTLY AGENT_EFF_MEAN in every simulation.
    The half-range is small enough that centering cannot leave [0, 1].
    """
    offsets = rng.uniform(
        -AGENT_EFF_HALF_RANGE,
        AGENT_EFF_HALF_RANGE,
        size=(N_AGENTS, N_CLIENTS),
    )
    offsets -= offsets.mean()
    eff = AGENT_EFF_MEAN + offsets

    # Numerical guard only; by construction values should be safely in [0, 1].
    if not np.all((eff >= 0.0) & (eff <= 1.0)):
        raise RuntimeError("Agent effectiveness left [0, 1].")

    if not np.isclose(eff.mean(), AGENT_EFF_MEAN, atol=1e-12):
        raise RuntimeError("Agent matrix mean is not preserved.")

    return eff


# ============================================================
# Optimization helpers
# ============================================================

def score_matrix_for_hour(
    p_response: np.ndarray,
    agent_eff: np.ndarray,
    hour_idx: int,
    clients: np.ndarray,
) -> np.ndarray:
    """
    Matrix [agent, client] of expected success probabilities for one hour.
    """
    return agent_eff[:, clients] * p_response[clients, hour_idx][None, :]


def optimize_slots(
    p_response: np.ndarray,
    agent_eff: np.ndarray,
    clients: Iterable[int],
    start_hour_idx: int,
) -> List[Assignment]:
    """
    Solve maximum-weight assignment over all slots from start_hour_idx to end.

    Rows = (hour, agent) slots.
    Columns = eligible clients.
    linear_sum_assignment minimizes, so we pass negative expected value.
    """
    clients = np.array(sorted(clients), dtype=int)

    slots: List[Tuple[int, int]] = [
        (hour_idx, agent)
        for hour_idx in range(start_hour_idx, N_HOURS)
        for agent in range(N_AGENTS)
    ]

    weights = np.zeros((len(slots), len(clients)), dtype=float)

    for row, (hour_idx, agent) in enumerate(slots):
        weights[row, :] = (
            p_response[clients, hour_idx] * agent_eff[agent, clients]
        )

    row_idx, col_idx = linear_sum_assignment(-weights)

    return [
        Assignment(
            hour_idx=slots[row][0],
            agent=slots[row][1],
            client=int(clients[col]),
        )
        for row, col in zip(row_idx, col_idx)
    ]


# ============================================================
# Policies
# ============================================================

def policy_random(
    rng: np.random.Generator,
) -> List[Assignment]:
    """
    Randomly choose 4 not-yet-called clients each hour and randomly assign them.
    """
    remaining = list(range(N_CLIENTS))
    schedule: List[Assignment] = []

    for hour_idx in range(N_HOURS):
        chosen = rng.choice(remaining, size=N_AGENTS, replace=False).tolist()
        agent_order = rng.permutation(N_AGENTS)

        for agent, client in zip(agent_order, chosen):
            schedule.append(Assignment(hour_idx, int(agent), int(client)))

        chosen_set = set(chosen)
        remaining = [c for c in remaining if c not in chosen_set]

    return schedule


def policy_carousel(
    p_response: np.ndarray,
) -> List[Assignment]:
    """
    Baseline carousel:
    - every hour rank remaining clients by CURRENT response probability;
    - take top 4;
    - give them to agents 0,1,2,3 without agent-client optimization.
    """
    remaining = set(range(N_CLIENTS))
    schedule: List[Assignment] = []

    for hour_idx in range(N_HOURS):
        ranked = sorted(
            remaining,
            key=lambda c: p_response[c, hour_idx],
            reverse=True,
        )
        chosen = ranked[:N_AGENTS]

        for agent, client in enumerate(chosen):
            schedule.append(Assignment(hour_idx, agent, client))

        remaining.difference_update(chosen)

    return schedule


def policy_hourly_greedy(
    p_response: np.ndarray,
    agent_eff: np.ndarray,
) -> List[Assignment]:
    """
    Agent-aware greedy:
    maximize current-hour q = response probability * agent effectiveness.
    It ignores all future hours.
    """
    remaining = set(range(N_CLIENTS))
    schedule: List[Assignment] = []

    for hour_idx in range(N_HOURS):
        clients = np.array(sorted(remaining), dtype=int)
        weights = score_matrix_for_hour(
            p_response, agent_eff, hour_idx, clients
        )

        agent_rows, client_cols = linear_sum_assignment(-weights)

        called_now = []
        for agent, col in zip(agent_rows, client_cols):
            client = int(clients[col])
            schedule.append(
                Assignment(hour_idx, int(agent), client)
            )
            called_now.append(client)

        remaining.difference_update(called_now)

    return schedule


def policy_full_horizon(
    p_response: np.ndarray,
    agent_eff: np.ndarray,
) -> List[Assignment]:
    """
    Optimize all 16 slots jointly at the beginning of the day.
    """
    return optimize_slots(
        p_response=p_response,
        agent_eff=agent_eff,
        clients=range(N_CLIENTS),
        start_hour_idx=0,
    )


def policy_rolling_horizon(
    p_response: np.ndarray,
    agent_eff: np.ndarray,
) -> List[Assignment]:
    """
    Re-optimize all remaining hours at the beginning of every hour,
    execute only the current-hour assignments, then repeat.

    With static probabilities and no new information this should be very close
    to the full-horizon optimum. It becomes materially different once forecasts
    or constraints update during the day.
    """
    remaining = set(range(N_CLIENTS))
    schedule: List[Assignment] = []

    for current_hour_idx in range(N_HOURS):
        plan = optimize_slots(
            p_response=p_response,
            agent_eff=agent_eff,
            clients=remaining,
            start_hour_idx=current_hour_idx,
        )

        execute_now = [
            x for x in plan if x.hour_idx == current_hour_idx
        ]
        schedule.extend(execute_now)

        remaining.difference_update(x.client for x in execute_now)

    return schedule


# ============================================================
# Monte-Carlo engine
# ============================================================

def simulate_schedule(
    schedule: List[Assignment],
    p_response: np.ndarray,
    agent_eff: np.ndarray,
    common_uniforms: np.ndarray,
) -> List[dict]:
    """
    Simulate actual collection results for a schedule.

    common_uniforms[hour, agent, client] is shared across policies inside a
    simulation. This is a common-random-numbers variance-reduction technique.
    """
    rows = []

    for x in schedule:
        q = (
            p_response[x.client, x.hour_idx]
            * agent_eff[x.agent, x.client]
        )

        success = (
            common_uniforms[x.hour_idx, x.agent, x.client] < q
        )

        rows.append({
            "hour_idx": x.hour_idx,
            "hour": int(HOURS[x.hour_idx]),
            "agent": x.agent + 1,
            "client": x.client + 1,
            "p_response": float(p_response[x.client, x.hour_idx]),
            "agent_effectiveness": float(agent_eff[x.agent, x.client]),
            "success_probability": float(q),
            "success": int(success),
            "collected_amount": float(success) * DEBT_PER_CLIENT,
        })

    return rows


def run_experiment(
    seed: int = SEED,
    n_simulations: int = N_SIMULATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      calls_df   - one row per attempted call
      hourly_df  - one row per simulation x policy x hour
      daily_df   - one row per simulation x policy
    """
    rng = np.random.default_rng(seed)

    # Fixed client-hour probabilities across all simulations.
    p_response = make_response_probabilities(rng)

    # Carousel does not depend on daily agent matrix.
    fixed_carousel = policy_carousel(p_response)

    call_records: List[dict] = []

    for sim in range(n_simulations):
        agent_eff = make_agent_effectiveness(rng)

        # Common random numbers for fairer policy comparisons.
        common_uniforms = rng.random(
            size=(N_HOURS, N_AGENTS, N_CLIENTS)
        )

        schedules: Dict[str, List[Assignment]] = {
            "Random": policy_random(rng),
            "Carousel": fixed_carousel,
            "Hourly greedy": policy_hourly_greedy(
                p_response, agent_eff
            ),
            "Full-horizon optimum": policy_full_horizon(
                p_response, agent_eff
            ),
            "Rolling horizon": policy_rolling_horizon(
                p_response, agent_eff
            ),
        }

        for policy, schedule in schedules.items():
            simulated_rows = simulate_schedule(
                schedule=schedule,
                p_response=p_response,
                agent_eff=agent_eff,
                common_uniforms=common_uniforms,
            )

            for row in simulated_rows:
                row["simulation"] = sim
                row["policy"] = policy
                call_records.append(row)

    calls_df = pd.DataFrame(call_records)

    hourly_df = (
        calls_df
        .groupby(
            ["simulation", "policy", "hour"],
            as_index=False,
        )
        .agg(
            attempts=("success", "size"),
            collected_clients=("success", "sum"),
            collected_amount=("collected_amount", "sum"),
            expected_collected_clients=("success_probability", "sum"),
            mean_call_success_probability=("success_probability", "mean"),
        )
    )

    daily_df = (
        hourly_df
        .groupby(
            ["simulation", "policy"],
            as_index=False,
        )
        .agg(
            collected_clients=("collected_clients", "sum"),
            collected_amount=("collected_amount", "sum"),
            expected_collected_clients=("expected_collected_clients", "sum"),
        )
    )

    return calls_df, hourly_df, daily_df


# ============================================================
# Reporting
# ============================================================

def make_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    return (
        daily_df
        .groupby("policy")
        .agg(
            mean_actual_clients=("collected_clients", "mean"),
            std_actual_clients=("collected_clients", "std"),
            p10_actual_clients=(
                "collected_clients",
                lambda s: np.quantile(s, 0.10),
            ),
            median_actual_clients=("collected_clients", "median"),
            p90_actual_clients=(
                "collected_clients",
                lambda s: np.quantile(s, 0.90),
            ),
            mean_expected_clients=("expected_collected_clients", "mean"),
            mean_collected_amount=("collected_amount", "mean"),
        )
        .reindex(POLICY_ORDER)
        .reset_index()
    )


def make_hourly_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    return (
        hourly_df
        .groupby(["policy", "hour"])
        .agg(
            mean_actual_clients=("collected_clients", "mean"),
            mean_expected_clients=("expected_collected_clients", "mean"),
            mean_collected_amount=("collected_amount", "mean"),
        )
        .reset_index()
    )


def plot_daily_distribution(
    daily_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    max_success = int(daily_df["collected_clients"].max())
    bins = np.arange(-0.5, max_success + 1.5, 1)

    for policy in POLICY_ORDER:
        values = daily_df.loc[
            daily_df["policy"] == policy,
            "collected_clients",
        ]
        plt.hist(
            values,
            bins=bins,
            alpha=0.35,
            label=policy,
        )

    plt.xlabel("Successful collections during the day")
    plt.ylabel("Number of Monte-Carlo simulations")
    plt.title("Daily result distribution by policy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_hourly_distribution(
    hourly_df: pd.DataFrame,
    output_path: Path,
) -> None:
    data = []
    labels = []

    for hour in HOURS:
        for policy in POLICY_ORDER:
            values = hourly_df.loc[
                (hourly_df["hour"] == hour)
                & (hourly_df["policy"] == policy),
                "collected_clients",
            ].to_numpy()

            data.append(values)
            labels.append(f"{hour}:00\n{policy}")

    plt.figure(figsize=(17, 7))
    plt.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )
    plt.ylabel("Successful collections in the hour")
    plt.title(
        f"Hourly result distribution; {N_SIMULATIONS} simulations per hour"
    )
    plt.xticks(rotation=55, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_expected_vs_actual(
    daily_df: pd.DataFrame,
    output_path: Path,
) -> None:
    summary = (
        daily_df
        .groupby("policy")
        .agg(
            actual=("collected_clients", "mean"),
            expected=("expected_collected_clients", "mean"),
        )
        .reindex(POLICY_ORDER)
    )

    x = np.arange(len(summary))
    width = 0.38

    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        summary["actual"],
        width=width,
        label="Actual Monte-Carlo mean",
    )
    plt.bar(
        x + width / 2,
        summary["expected"],
        width=width,
        label="Expected value",
    )
    plt.xticks(x, summary.index, rotation=25, ha="right")
    plt.ylabel("Collections per day")
    plt.title("Expected vs actual mean result")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def validate_results(
    calls_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
) -> None:
    """
    Basic structural checks.
    """
    # Exactly four calls per policy per hour per simulation.
    if not (hourly_df["attempts"] == N_AGENTS).all():
        raise AssertionError("Not every policy uses all 4 agents every hour.")

    # At most one call to each client within simulation-policy.
    max_calls = (
        calls_df
        .groupby(["simulation", "policy", "client"])
        .size()
        .max()
    )
    if max_calls > 1:
        raise AssertionError("A client was called more than once in a day.")

    # Exactly 16 total calls per policy-day.
    daily_attempts = (
        calls_df
        .groupby(["simulation", "policy"])
        .size()
    )
    if not (daily_attempts == N_AGENTS * N_HOURS).all():
        raise AssertionError("Policy-day does not have exactly 16 calls.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calls_df, hourly_df, daily_df = run_experiment()
    validate_results(calls_df, hourly_df)

    summary_df = make_summary(daily_df)
    hourly_summary_df = make_hourly_summary(hourly_df)

    # Save data.
    calls_df.to_csv(
        OUTPUT_DIR / "calls.csv",
        index=False,
    )
    hourly_df.to_csv(
        OUTPUT_DIR / "hourly_results.csv",
        index=False,
    )
    daily_df.to_csv(
        OUTPUT_DIR / "daily_results.csv",
        index=False,
    )
    summary_df.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )
    hourly_summary_df.to_csv(
        OUTPUT_DIR / "hourly_summary.csv",
        index=False,
    )

    # Save the fixed response-probability matrix separately.
    rng_for_static_data = np.random.default_rng(SEED)
    p_response = make_response_probabilities(rng_for_static_data)
    p_df = pd.DataFrame(
        p_response,
        columns=[f"{h}:00" for h in HOURS],
    )
    p_df.insert(
        0,
        "client",
        np.arange(1, N_CLIENTS + 1),
    )
    p_df.to_csv(
        OUTPUT_DIR / "response_probabilities.csv",
        index=False,
    )

    # Plots.
    plot_daily_distribution(
        daily_df,
        OUTPUT_DIR / "daily_distribution.png",
    )
    plot_hourly_distribution(
        hourly_df,
        OUTPUT_DIR / "hourly_distribution.png",
    )
    plot_expected_vs_actual(
        daily_df,
        OUTPUT_DIR / "expected_vs_actual.png",
    )

    print("\n=== DAILY SUMMARY ===")
    print(
        summary_df.round(3).to_string(index=False)
    )

    print("\n=== HOURLY MEAN ACTUAL COLLECTIONS ===")
    hourly_pivot = (
        hourly_summary_df
        .pivot(
            index="hour",
            columns="policy",
            values="mean_actual_clients",
        )
        .reindex(columns=POLICY_ORDER)
    )
    print(hourly_pivot.round(3).to_string())

    print("\n=== RESPONSE PROBABILITIES, CLIENT 1 ===")
    print(
        p_df.loc[p_df["client"] == 1]
        .round(3)
        .to_string(index=False)
    )

    print(f"\nFiles saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
