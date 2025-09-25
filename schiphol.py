import datetime as dt
import time
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import matplotlib as plt


# ====== Config ======
APP_ID   = "3a9d1776"
APP_KEY  = "a80798c35e9852feaeacb7660bc130b0"
BASE_URL = "https://api.schiphol.nl/public-flights/flights"

# Kies ophaalmethode
USE_WINDOW = True          # True = tijdvenster (24u terug, 6u vooruit). False = per dag.
N_DAYS     = 7             # Alleen gebruikt als USE_WINDOW=False
DIRECTIONS = ["D", "A"]    # "D" = departures, "A" = arrivals

# Veelvoorkomende airlines op/naar AMS: IATA -> naam
AIRLINE_MAP = {
    "KL": "KLM", "HV": "Transavia", "U2": "easyJet", "EJU": "easyJet Europe",
    "LH": "Lufthansa", "AF": "Air France", "BA": "British Airways", "LX": "SWISS",
    "OS": "Austrian", "SK": "SAS", "AY": "Finnair", "IB": "Iberia", "VY": "Vueling",
    "UX": "Air Europa", "TP": "TAP Air Portugal", "SN": "Brussels Airlines",
    "FR": "Ryanair", "W6": "Wizz Air", "PC": "Pegasus", "TK": "Turkish Airlines",
    "QR": "Qatar Airways", "EK": "Emirates", "ET": "Ethiopian Airlines",
    "SQ": "Singapore Airlines", "CX": "Cathay Pacific", "CI": "China Airlines",
    "KE": "Korean Air", "NH": "ANA", "JL": "Japan Airlines", "VS": "Virgin Atlantic",
    "DL": "Delta Air Lines", "UA": "United Airlines", "AA": "American Airlines",
    "AC": "Air Canada", "OR": "TUI fly Netherlands", "QY": "DHL (EAT Leipzig)",
    "MP": "Martinair Cargo"
}

# ====== Tijdvenster (laatste 24 uur t/m 6 uur vooruit) ======
now = dt.datetime.now(dt.UTC)
time_from = (now - dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
time_to   = (now + dt.timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S")

headers = {
    "ResourceVersion": "v4",
    "Accept": "application/json",
    "app_id": APP_ID,
    "app_key": APP_KEY,
}

# Basis params voor window-mode
params_window = {
    "includedelays": "true",
    "page": 0,
    "sort": "+scheduleTime",
    "fromDateTime": time_from,
    "toDateTime": time_to,
    # "flightDirection": "A" / "D" wordt in de fetch-functie gezet
}

# === HTTP helper met backoff ===
def get_with_retries(url, headers, params, tries=8, timeout=30):
    """HTTP GET met exponential backoff; respecteert Retry-After bij 429."""
    wait = 0.5
    for attempt in range(1, tries + 1):
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 200:
            return r

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    sleep_s = float(ra)
                except ValueError:
                    sleep_s = wait
            else:
                sleep_s = wait
            print(f"[429] rate-limited op page={params.get('page')} → slapen {sleep_s:.1f}s (poging {attempt}/{tries})")
            time.sleep(sleep_s)
            wait = min(wait * 2, 16)
            continue

        if r.status_code in (500, 502, 503, 504):
            print(f"[{r.status_code}] server error op page={params.get('page')} → slapen {wait:.1f}s (poging {attempt}/{tries})")
            time.sleep(wait)
            wait = min(wait * 2, 16)
            continue

        raise RuntimeError(f"HTTP {r.status_code} voor params={params} :: {r.text[:300]}")
    raise RuntimeError(f"Failed after retries: {params}")

# === Ophalen binnen tijdvenster ===
def fetch_window(params_base: dict, directions=("D","A"), max_pages_per_dir=2000):
    """Haal alle pagina's op binnen een tijdvenster; stopt op links.next ontbreekt of lege page."""
    all_flights = []
    for direction in directions:
        p = params_base.copy()
        p["page"] = 0
        p["flightDirection"] = direction
        while p["page"] < max_pages_per_dir:
            try:
                resp = get_with_retries(BASE_URL, headers, p)
            except RuntimeError as e:
                print(f"[WARN] stop {direction} op page={p['page']} door error: {e}")
                break

            data = resp.json() or {}
            chunk = data.get("flights", [])
            if not chunk:
                print(f"[INFO] {direction}: lege page op {p['page']} → klaar.")
                break

            all_flights.extend(chunk)

            links = data.get("links", [])
            has_next = any(l.get("rel") == "next" for l in links)
            if not has_next:
                print(f"[INFO] {direction}: geen next-link na page {p['page']} → klaar.")
                break

            p["page"] += 1
            time.sleep(0.15)
    return all_flights

# === Ophalen per kalenderdag ===
def fetch_days(n_days: int, directions=("D","A")):
    """Haal per dag (scheduleDate) alle pagina's op voor n_days en richtingen."""
    all_flights = []
    dates = [dt.date.today() - dt.timedelta(days=i) for i in range(n_days)]
    for day in dates:
        for direction in directions:
            page = 0
            while True:
                p = {
                    "scheduleDate": str(day),
                    "flightDirection": direction,
                    "sort": "+scheduleTime",
                    "page": page
                }
                resp = get_with_retries(BASE_URL, headers, p)
                data = resp.json() or {}
                chunk = data.get("flights", [])
                if not chunk:
                    break
                all_flights.extend(chunk)
                page += 1
                time.sleep(0.12)
    return all_flights

# === Tijd casting helper ===
def ensure_series_datetime(df: pd.DataFrame, colname: str) -> pd.Series:
    """Altijd datetime64[ns, UTC] Series."""
    if colname not in df.columns:
        return pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[colname], errors="coerce", utc=True)

# === DataFrame bouwen + features ===
def build_dataframe(all_flights: list) -> pd.DataFrame:
    df = pd.json_normalize(all_flights)

    # Geplande tijd
    df["scheduled_dt"] = ensure_series_datetime(df, "scheduleDateTime")

    # ARRIVALS
    arr_actual   = ensure_series_datetime(df, "actualLandingTime")
    arr_est      = ensure_series_datetime(df, "estimatedLandingTime")
    arr_expected = ensure_series_datetime(df, "expectedTimeOnBelt")  # fallback
    arrival_best = arr_actual.fillna(arr_est).fillna(arr_expected)

    # DEPARTURES
    dep_actual   = ensure_series_datetime(df, "actualOffBlockTime")
    dep_est      = ensure_series_datetime(df, "estimatedOffBlockTime")
    dep_expected = ensure_series_datetime(df, "expectedTimeGate")    # fallback
    departure_best = dep_actual.fillna(dep_est).fillna(dep_expected)

    # Combineer arrivals/departures
    flight_dir = df.get("flightDirection")
    if flight_dir is None:
        flight_dir = pd.Series(["D"] * len(df), index=df.index)
    df["best_actual_dt"] = np.where(flight_dir.eq("A"), arrival_best, departure_best)

    # Vertraging in minuten
    diff = (df["best_actual_dt"] - df["scheduled_dt"])
    df["delay_minutes"] = (diff.dt.total_seconds() / 60).round(0)

    # Airline naam kolom
    if "prefixIATA" in df.columns:
        df["airline_name"] = df["prefixIATA"].map(AIRLINE_MAP).fillna(df["prefixIATA"])
    else:
        df["airline_name"] = "Onbekend"

    # Cancel-flag (CNX)
    def has_cnx(states):
        if isinstance(states, list):
            return any(s == "CNX" for s in states)
        if pd.isna(states):
            return False
        return "CNX" in str(states)

    if "publicFlightState.flightStates" in df.columns:
        df["is_cancelled"] = df["publicFlightState.flightStates"].apply(has_cnx)
    else:
        df["is_cancelled"] = False

    # Handige datum (Europe/Amsterdam)
    df["scheduled_local"] = df["scheduled_dt"].dt.tz_convert("Europe/Amsterdam")
    df["day"] = df["scheduled_local"].dt.date

    # --- Aircraft type (robuust opbouwen) ---
    def coalesce_cols(df, cols):
        s = pd.Series(pd.NA, index=df.index, dtype="object")
        for c in cols:
            if c in df.columns:
                s = s.fillna(df[c])
        return s

    df["aircraft_type"] = (
        coalesce_cols(df, [
            "aircraftType.iataMain", "aircraftType.iataSub",
            "aircraftType.icaoMain", "aircraftType.icaoSub",
            "aircraftType"  # fallback
        ])
        .astype(str).str.strip().str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )

    return df

# === Main ===
def main():
    print("Ophalen van flights...")
    if USE_WINDOW:
        flights = fetch_window(params_window, DIRECTIONS)
    else:
        flights = fetch_days(N_DAYS, DIRECTIONS)

    if not flights:
        print("Geen data ontvangen in dit venster.")
        return

    df = build_dataframe(flights)
    print(f"Totaal vluchten: {len(df)}  |  Unieke airlines: {df['airline_name'].nunique()}")

    # ====== Analyses / Visualisaties ======
    viz = df.dropna(subset=["delay_minutes"]).copy()

    # Plot 1: Histogram per airline
    if "airline_name" in viz.columns:
        fig1 = px.histogram(
            viz, x="delay_minutes", color="airline_name",
            nbins=40, barmode="overlay",
            title="Verdeling van vertragingen per airline"
        )
        fig1.update_layout(xaxis_title="Vertraging (minuten)", yaxis_title="Aantal vluchten")
        fig1.show()

    # Plot 2: Gemiddelde vertraging per airline
    if "airline_name" in viz.columns:
        df_airline = (
            viz.groupby("airline_name", dropna=False)["delay_minutes"]
               .mean().reset_index()
               .sort_values("delay_minutes", ascending=False)
        )
        fig2 = px.bar(
            df_airline, x="airline_name", y="delay_minutes",
            title="Gemiddelde vertraging per airline",
            labels={"airline_name": "Airline", "delay_minutes": "Gemiddelde vertraging (minuten)"}
        )
        fig2.update_layout(xaxis_tickangle=-45)
        fig2.show()

        delayed_flights = viz[viz["delay_minutes"] > 0]

        # Rate (>= 0 min) per airline
        totaal = viz.groupby("airline_name", dropna=False).size().rename("totaal")
        vertraagd = delayed_flights.groupby("airline_name", dropna=False).size().rename("vertraagd")
        rates = pd.concat([totaal, vertraagd], axis=1).fillna(0).reset_index()
        rates["vertraagd_pct"] = (rates["vertraagd"] / rates["totaal"] * 100).round(1)
        fig_rate = px.bar(
            rates.sort_values("vertraagd_pct", ascending=False),
            x="airline_name", y="vertraagd_pct",
            title="Vertraagde vluchten (%) per airline",
            labels={"airline_name": "Airline", "vertraagd_pct": "Vertraagd (%)"}
        )
        fig_rate.update_layout(xaxis_tickangle=-45)
        fig_rate.show()

        if not delayed_flights.empty:
            airline_counts = (delayed_flights.groupby("airline_name")
                              .size().reset_index(name="aantal_vertraagd")
                              .sort_values("aantal_vertraagd", ascending=False))
            fig3 = px.pie(
                airline_counts, values="aantal_vertraagd", names="airline_name",
                title="Percentage vertraagde vluchten per airline"
            )
            fig3.show()

    # ====== Cancellations per dag ======
    daily_cancel = (
        df.groupby("day")
          .agg(flights=("is_cancelled","size"), cancelled=("is_cancelled","sum"))
          .reset_index()
    )
    daily_cancel["cancel_rate_%"] = (daily_cancel["cancelled"] / daily_cancel["flights"] * 100).round(1)

    fig_rate = px.line(
        daily_cancel, x="day", y="cancel_rate_%", markers=True,
        title="Cancel-rate per dag (%)",
        labels={"day": "Dag", "cancel_rate_%": "Cancel-rate (%)"}
    )
    fig_rate.show()

    # ====== HISTOGRAM & PIE per vliegtuigtype ======
    vt = df.dropna(subset=["delay_minutes", "aircraft_type"]).copy()
    if vt.empty:
        print("Geen aircraft types + delay gevonden in deze periode.")
    else:
        # Histogram: vertraging per type (overlay). Beperk tot top-N types voor leesbaarheid.
        topN = 12
        top_types = vt["aircraft_type"].value_counts().head(topN).index.tolist()
        vt_top = vt[vt["aircraft_type"].isin(top_types)]

        fig_hist_type = px.histogram(
            vt_top,
            x="delay_minutes",
            color="aircraft_type",
            nbins=50,
            histnorm="percent",     # % binnen de hele selectie, handig om vormen te vergelijken
            barmode="overlay",
            opacity=0.55,
            title=f"Histogram vertraging per vliegtuigtype (top {topN})",
            labels={"delay_minutes": "Vertraging (min)", "aircraft_type": "Vliegtuigtype"}
        )
        fig_hist_type.update_layout(xaxis_tickangle=0, legend_title_text="Vliegtuigtype")
        fig_hist_type.show()

        # Pie: aandeel van vertraagde vluchten per type (>0 min)
        delayed = vt[vt["delay_minutes"] > 0].copy()
        if delayed.empty:
            print("Geen vertraagde vluchten (>0 min) voor de piechart.")
        else:
            counts = delayed["aircraft_type"].value_counts().reset_index()
            counts.columns = ["aircraft_type", "delayed_flights"]

            # Kleine slices samenvoegen in 'Overig' voor nette pie
            keep = counts.head(12)
            other_sum = counts["delayed_flights"][12:].sum()
            if other_sum > 0:
                keep = pd.concat(
                    [keep, pd.DataFrame([{"aircraft_type": "Overig", "delayed_flights": other_sum}])],
                    ignore_index=True
                )

            fig_pie_type = px.pie(
                keep,
                values="delayed_flights",
                names="aircraft_type",
                title="Vertraagde vluchten: verdeling per vliegtuigtype"
            )
            fig_pie_type.show()


if __name__ == "__main__":
    main()

