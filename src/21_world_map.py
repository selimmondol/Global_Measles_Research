
from pathlib import Path
import pandas as pd
import pycountry
import plotly.express as px

# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "validation_design"
    / "country_holdout.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "fig01_panelB_world_map.png"
OUTPUT_PDF = OUTPUT_DIR / "fig01_panelB_world_map.pdf"
OUTPUT_CSV = OUTPUT_DIR / "fig01_country_mapping_audit.csv"

# ============================================================
# 2. LOAD COUNTRY CLASSIFICATIONS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Country holdout file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {"Country", "holdout_status"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required_columns - set(df.columns)}"
    )

if df["Country"].duplicated().any():
    raise ValueError("Duplicate country names detected.")

# ============================================================
# 3. COUNTRY NAME TO ISO3 MAPPING
# ============================================================

country_fixes = {
    "Bolivia (Plurinational State of)": "BOL",
    "China, Macao SAR": "MAC",
    "Democratic People's Republic of Korea": "PRK",
    "Netherlands (Kingdom of the)": "NLD",
    "Republic of Korea": "KOR",
    "Democratic Republic of the Congo": "COD",
    "Micronesia (Federated States of)": "FSM",
    "Naoero": "NRU",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "China, Hong Kong SAR": "HKG",
    "Iran (Islamic Republic of)": "IRN",
    "occupied Palestinian territory, including east Jerusalem": "PSE",
}


def get_iso3(country_name):
    name = str(country_name).strip()

    if name in country_fixes:
        return country_fixes[name]

    if name.lower().startswith("occupied palestinian territory"):
        return "PSE"

    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


df["ISO3"] = df["Country"].apply(get_iso3)

# ============================================================
# 4. VALIDATE COUNTRY MATCHING
# ============================================================

unmatched = df[df["ISO3"].isna()]

print("\nCOUNTRY MAPPING AUDIT")
print("-" * 45)
print("Total classified countries:", len(df))
print("ISO3 matched:", df["ISO3"].notna().sum())
print("ISO3 unmatched:", len(unmatched))

if not unmatched.empty:
    print("\nUnmatched countries:")
    print(unmatched["Country"].to_string(index=False))

    raise ValueError(
        "Resolve unmatched country names before creating map."
    )

if df["ISO3"].duplicated().any():
    print(df[df["ISO3"].duplicated(False)])
    raise ValueError("Duplicate ISO3 codes detected.")

# ============================================================
# 5. DEFINE COUNTRY GROUPS
# ============================================================

group_names = {
    "development_country": "Development countries",
    "unseen_country": "Held-out countries",
}

df["Group"] = df["holdout_status"].map(group_names)

if df["Group"].isna().any():
    raise ValueError("Unknown holdout_status detected.")

development_count = (
    df["holdout_status"] == "development_country"
).sum()

heldout_count = (
    df["holdout_status"] == "unseen_country"
).sum()

print("\nCOUNTRY GROUPS")
print("-" * 45)
print("Development countries:", development_count)
print("Held-out countries:", heldout_count)

if development_count != 162 or heldout_count != 40:
    raise ValueError(
        "Country counts do not match the frozen study design."
    )

# ============================================================
# 6. SAVE MAPPING AUDIT
# ============================================================

df[
    ["Country", "ISO3", "holdout_status", "Group"]
].to_csv(OUTPUT_CSV, index=False)

# ============================================================
# 7. CREATE WORLD MAP
# ============================================================

fig = px.choropleth(
    df,
    locations="ISO3",
    locationmode="ISO-3",
    color="Group",
    hover_name="Country",
    category_orders={
        "Group": [
            "Development countries",
            "Held-out countries",
        ]
    },
    color_discrete_map={
        "Development countries": "#4778A8",
        "Held-out countries": "#D94E4E",
    },
)

# ============================================================
# 8. PUBLICATION-STYLE GEOGRAPHICAL SETTINGS
# ============================================================

fig.update_geos(
    projection_type="natural earth",
    showframe=False,
    showcoastlines=True,
    coastlinecolor="#B8C0C9",
    coastlinewidth=0.4,
    showcountries=True,
    countrycolor="#FFFFFF",
    countrywidth=0.5,
    showland=True,
    landcolor="#E7EBF0",
    showocean=True,
    oceancolor="#FFFFFF",
    bgcolor="#FFFFFF",
)

fig.update_traces(
    marker_line_color="#FFFFFF",
    marker_line_width=0.4,
)

# ============================================================
# 9. FINAL LAYOUT
# ============================================================

fig.update_layout(
    title=None,
    width=1600,
    height=850,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(
        family="Arial",
        size=19,
        color="#253348",
    ),
    legend=dict(
        title=None,
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.02,
        yanchor="bottom",
        font=dict(size=19),
    ),
)

# ============================================================
# 10. EXPORT FIGURES
# ============================================================

fig.write_image(
    str(OUTPUT_PNG),
    width=1600,
    height=850,
    scale=2,
)

fig.write_image(
    str(OUTPUT_PDF),
    width=1600,
    height=850,
)

print("\nFIGURE EXPORT COMPLETED")
print("-" * 45)
print("PNG:", OUTPUT_PNG)
print("PDF:", OUTPUT_PDF)
print("Country audit:", OUTPUT_CSV)

print("\nNOTE:")
print(
    "Small island territories may not be visible "
    "at the scale of the world map."
)

print(
    "ISO3 validation does not guarantee that "
    "every territory is displayed by Plotly."
)
