import streamlit as st
import pandas as pd
import json, os

# --- Page config: wide & title ---
st.set_page_config(page_title="Fantasy Football Player Rankings — Gridiron AI", layout="wide")

# ---- Display config ----
TABLE_HEIGHT = 950  # taller, scrollable table

# ---- Dataset column names (as they exist in CSV) ----
VALUE_COL = "Value (How valuable a player is overall taking into account both projected points and how difficult it is to find a replacement player at their position.)"
EXPECTED_COL = "Expected Points (Expected fantasy points based on the selected scoring settings.)"
DOWNSIDE_COL = "Downside Points (Downside fantasy points based on the selected scoring settings.)"
UPSIDE_COL = "Upside Points (Upside fantasy points based on the selected scoring settings.)"

STATE_PATH = "data/draft_state.json"

# ---- Persistence helpers ----
def save_state(df, history):
    """Persist minimal state to disk: Drafted/MyTeam per player (by Name) and undo history by Name."""
    try:
        payload = {
            "rows": df[["Name", "Drafted", "MyTeam"]].to_dict(orient="records"),
            "history": history,  # list of {"name": str, "prev": {"Drafted": bool, "MyTeam": bool}}
        }
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save state: {e}")

def load_state():
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load saved state: {e}")
        return None

def apply_state_to_df(df, state):
    if not state or "rows" not in state:
        return df
    saved = {row["Name"]: row for row in state["rows"]}
    for idx, name in df["Name"].items():
        if name in saved:
            df.at[idx, "Drafted"] = bool(saved[name].get("Drafted", False))
            df.at[idx, "MyTeam"] = bool(saved[name].get("MyTeam", False))
    return df

# ---- Load data ----
@st.cache_data
def load_data():
    df = pd.read_csv("data/gridironai-rankings-2025-0.csv")

    # Ensure numerics for filtering/formatting
    def to_num(col):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for c in [VALUE_COL, EXPECTED_COL, DOWNSIDE_COL, UPSIDE_COL]:
        to_num(c)

    # Status flags
    if "Drafted" not in df.columns:
        df["Drafted"] = False              # drafted by anyone
    if "MyTeam" not in df.columns:
        df["MyTeam"] = False               # drafted by Eddie

    return df

# ---- Session state bootstrap ----
if "data" not in st.session_state:
    base_df = load_data()
    persisted = load_state()
    base_df = apply_state_to_df(base_df, persisted)
    st.session_state["data"] = base_df

if "history" not in st.session_state:
    persisted = load_state()
    st.session_state["history"] = persisted["history"] if (persisted and "history" in persisted) else []

df = st.session_state["data"]

st.title("Fantasy Football Player Rankings — Gridiron AI")

# ---- Undo helpers (history stored by Name to avoid index drift) ----
def record_and_apply(idx: int, drafted: bool, myteam: bool):
    name = df.at[idx, "Name"]
    prev = {"Drafted": bool(df.at[idx, "Drafted"]), "MyTeam": bool(df.at[idx, "MyTeam"])}
    st.session_state["history"].append({"name": name, "prev": prev})
    df.at[idx, "Drafted"] = drafted
    df.at[idx, "MyTeam"] = myteam
    save_state(df, st.session_state["history"])

def undo_last():
    if not st.session_state["history"]:
        return
    last = st.session_state["history"].pop()
    name, prev = last["name"], last["prev"]
    match = df.index[df["Name"] == name]
    if len(match) > 0:
        idx = match[0]
        df.at[idx, "Drafted"] = prev["Drafted"]
        df.at[idx, "MyTeam"] = prev["MyTeam"]
    save_state(df, st.session_state["history"])

# ---- Sidebar filters ----
positions = st.sidebar.multiselect("Filter by Position", options=sorted(df["Pos."].dropna().unique().tolist()))
team = st.sidebar.selectbox("Filter by Team", options=["All"] + sorted(df["Team"].dropna().unique().tolist()))
search_name = st.sidebar.text_input("Search Player")

# --- Value filter (bar + type, integers only) ---
min_val = int(df[VALUE_COL].min(skipna=True)) if VALUE_COL in df.columns and df[VALUE_COL].notna().any() else 0
max_val = int(df[VALUE_COL].max(skipna=True)) if VALUE_COL in df.columns and df[VALUE_COL].notna().any() else 0

slider_min, slider_max = st.sidebar.slider(
    "Filter by Value (bar)",
    min_value=min_val, max_value=max_val,
    value=(min_val, max_val), step=1,
)
col_a, col_b = st.sidebar.columns(2)
with col_a:
    typed_min = st.number_input("Min (type)", min_value=min_val, max_value=max_val,
                                value=slider_min, step=1, key="typed_min")
with col_b:
    typed_max = st.number_input("Max (type)", min_value=min_val, max_value=max_val,
                                value=slider_max, step=1, key="typed_max")
min_choice = max(min_val, min(int(typed_min), int(typed_max)))
max_choice = min(max_val, max(int(typed_min), int(typed_max)))

# ---- Global Undo ----
undo_col1, _ = st.columns([1, 5])
with undo_col1:
    if st.button("Undo Last"):
        undo_last()
        st.rerun()

# ---- Tabs ----
tab_board, tab_myteam, tab_other = st.tabs(["Draft Board", "Eddie's Team", "Other Teams"])

with tab_board:
    # Apply filters
    filtered_df = df.copy()
    if positions:
        filtered_df = filtered_df[filtered_df["Pos."].isin(positions)]
    if team != "All":
        filtered_df = filtered_df[filtered_df["Team"] == team]
    if search_name:
        filtered_df = filtered_df[filtered_df["Name"].str.contains(search_name, case=False, na=False)]

    # Value range
    filtered_df = filtered_df[
        (filtered_df[VALUE_COL].fillna(min_val) >= min_choice) &
        (filtered_df[VALUE_COL].fillna(max_val) <= max_choice)
    ]

    # Show only available
    available_df = filtered_df[~filtered_df["Drafted"]].copy()

    st.subheader("Available Players")
    if available_df.empty:
        st.info("No players match the current filters.")
    else:
        # sort by Value low -> high
        available_df = available_df.sort_values(by=VALUE_COL, ascending=True)

        # Table columns: Player | Value | Expected | Downside | Upside | Team | Pos | Bye | Draft | Remove
        view_cols = ["Name", VALUE_COL, EXPECTED_COL, DOWNSIDE_COL, UPSIDE_COL, "Team", "Pos.", "Bye"]
        present_cols = [c for c in view_cols if c in available_df.columns]
        work = available_df[present_cols].copy()

        # Display as integers for consistency
        for c in [VALUE_COL, EXPECTED_COL, DOWNSIDE_COL, UPSIDE_COL]:
            if c in work.columns:
                work[c] = work[c].fillna(0).astype(int)

        work["Draft"] = False
        work["Remove"] = False
        work.index = available_df.index

        edited = st.data_editor(
            work,
            use_container_width=True,
            hide_index=True,
            height=TABLE_HEIGHT,
            column_order=["Draft", "Name", "Pos.", VALUE_COL, "Team", EXPECTED_COL, DOWNSIDE_COL, UPSIDE_COL, "Bye", "Remove"],
            column_config={
                "Name": st.column_config.TextColumn("Player", disabled=True),
                VALUE_COL: st.column_config.NumberColumn("Value", format="%d", disabled=True),
                EXPECTED_COL: st.column_config.NumberColumn("Expected Points", format="%d", disabled=True),
                DOWNSIDE_COL: st.column_config.NumberColumn("Downside Points", format="%d", disabled=True),
                UPSIDE_COL: st.column_config.NumberColumn("Upside Points", format="%d", disabled=True),
                "Team": st.column_config.TextColumn("Team", disabled=True),
                "Pos.": st.column_config.TextColumn("Pos", disabled=True),
                "Bye": st.column_config.NumberColumn("Bye", format="%d", disabled=True),
                "Draft": st.column_config.CheckboxColumn("Draft"),
                "Remove": st.column_config.CheckboxColumn("Remove"),
            },
        )

        to_draft = edited[edited["Draft"] == True]
        to_remove = edited[(edited["Remove"] == True) & (edited["Draft"] == False)]
        any_action = False
        for idx in to_draft.index:
            record_and_apply(idx, drafted=True, myteam=True); any_action = True
        for idx in to_remove.index:
            record_and_apply(idx, drafted=True, myteam=False); any_action = True
        if any_action:
            st.rerun()

# ---------- Roster allocation ----------
STARTING_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "TE", "FLEX"]
STARTING_COUNTS = {"QB":1, "RB":2, "WR":2, "TE":1, "FLEX":1}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}
BENCH_SLOTS = 7

def allocate_roster(my_team_df: pd.DataFrame):
    """Auto-assign starters (best value first), then bench."""
    mt = my_team_df.copy()
    mt[VALUE_COL] = mt[VALUE_COL].fillna(0).astype(int)
    mt = mt.sort_values(by=VALUE_COL, ascending=False)

    used = set()
    slots = {"QB": [], "RB": [], "WR": [], "TE": [], "FLEX": []}

    def take_best(pos_set, count):
        taken = []
        for idx, row in mt.iterrows():
            if len(taken) >= count: break
            if idx in used: continue
            if str(row.get("Pos.", "")) in pos_set:
                taken.append(idx); used.add(idx)
        return taken

    slots["QB"] = take_best({"QB"}, 1)
    slots["RB"] = take_best({"RB"}, 2)
    slots["WR"] = take_best({"WR"}, 2)
    slots["TE"] = take_best({"TE"}, 1)
    slots["FLEX"] = take_best(FLEX_ELIGIBLE, 1)

    remaining = [idx for idx in mt.index if idx not in used]
    bench = remaining[:BENCH_SLOTS]
    overflow = remaining[BENCH_SLOTS:]
    return slots, bench, overflow

def player_line(row):
    """Format: Player Name | Value: X — Team — Pos | Bye: X"""
    val = int(row[VALUE_COL]) if pd.notna(row[VALUE_COL]) else 0
    return f"**{row['Name']}** | **Value:** {val} — {row['Team']} — {row['Pos.']} | Bye: {row.get('Bye','—')}"

def slot_row(label, idx_or_none, undo_key_prefix):
    """Render a single slot with bullet for empty and Undo button when filled."""
    st.markdown(f"**{label}**")
    if idx_or_none is None:
        st.markdown("<div style='padding-left:8px;'>•</div>", unsafe_allow_html=True)
    else:
        row = df.loc[idx_or_none]
        st.write(player_line(row))
        if st.button("Undo Draft", key=f"{undo_key_prefix}_{idx_or_none}"):
            record_and_apply(idx_or_none, drafted=False, myteam=False)
            st.rerun()

with tab_myteam:
    st.subheader("Eddie's Team")

    my_team = df[(df["Drafted"]) & (df["MyTeam"])].copy()
    if my_team.empty:
        st.info("You haven't drafted anyone yet.")
    else:
        slots, bench, overflow = allocate_roster(my_team)

        qb_idx = slots["QB"][0] if slots["QB"] else None
        rb1_idx = slots["RB"][0] if len(slots["RB"]) > 0 else None
        rb2_idx = slots["RB"][1] if len(slots["RB"]) > 1 else None
        wr1_idx = slots["WR"][0] if len(slots["WR"]) > 0 else None
        wr2_idx = slots["WR"][1] if len(slots["WR"]) > 1 else None
        te_idx  = slots["TE"][0] if slots["TE"] else None
        flex_idx = slots["FLEX"][0] if slots["FLEX"] else None

        slot_row("QB:", qb_idx, "undo_qb")
        slot_row("RB", rb1_idx, "undo_rb1")
        slot_row("RB", rb2_idx, "undo_rb2")
        slot_row("WR", wr1_idx, "undo_wr1")
        slot_row("WR", wr2_idx, "undo_wr2")
        slot_row("TE", te_idx, "undo_te")

        st.markdown("**FLEX** _(RB/WR/TE)_")
        if flex_idx is None:
            st.markdown("<div style='padding-left:8px;'>•</div>", unsafe_allow_html=True)
        else:
            row = df.loc[flex_idx]
            st.write(player_line(row))
            if st.button("Undo Draft", key=f"undo_flex_{flex_idx}"):
                record_and_apply(flex_idx, drafted=False, myteam=False)
                st.rerun()

        st.markdown("### Bench")
        bench = bench[:BENCH_SLOTS]
        for i in range(BENCH_SLOTS):
            label = f"Slot {i+1}"
            if i < len(bench):
                bidx = bench[i]
                st.markdown(f"**{label}**")
                st.write(player_line(df.loc[bidx]))
                if st.button("Undo Draft", key=f"undo_bench_{bidx}"):
                    record_and_apply(bidx, drafted=False, myteam=False)
                    st.rerun()
            else:
                st.markdown(f"**{label}**")
                st.markdown("<div style='padding-left:8px;'>•</div>", unsafe_allow_html=True)

        if overflow:
            st.warning(f"{len(overflow)} additional drafted player(s) not shown (bench full).")

with tab_other:
    st.subheader("Other Teams (Removed from Board)")
    removed = df[(df["Drafted"]) & (~df["MyTeam"])].copy()
    if removed.empty:
        st.info("No players have been removed by other teams.")
    else:
        removed = removed.sort_values(by=VALUE_COL, ascending=True)
        removed[VALUE_COL] = removed[VALUE_COL].fillna(0).astype(int)
        for idx, row in removed.iterrows():
            col1, col2 = st.columns([6, 1.7])
            with col1:
                st.write(player_line(row))
            with col2:
                if st.button("Undo Remove", key=f"undo_other_{idx}"):
                    record_and_apply(idx, drafted=False, myteam=False)
                    st.rerun()
