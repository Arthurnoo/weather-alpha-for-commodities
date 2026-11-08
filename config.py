"""
Single source of truth for the project: universe, zones, seasons, dates and paths.

Every other module imports from here, so changing a zone or a date only
requires an edit in this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"              # untouched downloads
DATA_PROCESSED = ROOT / "data" / "processed"  # cleaned tables used by the notebook
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"
# Weather starts 10 years earlier: anomalies of year Y are measured against the
# climatology of years before Y only, so 2000 needs a 1990s history
WEATHER_START = "1990-01-01"
FIRST_TEST_YEAR = 2010  # models are trained on 2000-2009, then refit every January
WEEK_RULE = "W-FRI"     # weekly bars end on Friday (futures close of the week)

# ---------------------------------------------------------------------------
# Weather download boxes (lat_min, lat_max, lon_min, lon_max), lon in [-180, 180]
# Boxes are deliberately wide: the production weights decide what matters.
# ---------------------------------------------------------------------------
BOXES = {
    "north_america": (24.0, 50.0, -125.0, -66.0),  # US crops + gas demand
    "south_america": (-40.0, 13.0, -80.0, -34.0),  # Brazil, Argentina, Colombia
    "west_africa":   (3.0, 14.0, -10.0, 16.0),     # cocoa belt
    "europe_black_sea": (42.0, 56.0, -5.0, 50.0),  # EU + Ukraine + south Russia
    "south_asia":    (5.0, 35.0, 68.0, 106.0),     # India, Thailand (sugar)
}

# CPC Global Unified daily gauge-based grids (0.5 deg), served by NOAA PSL
CPC_VARIABLES = {
    "tmax": "cpc_global_temp/tmax",       # daily max temperature, deg C
    "tmin": "cpc_global_temp/tmin",       # daily min temperature, deg C
    "precip": "cpc_global_precip/precip", # daily precipitation, mm
}

# ---------------------------------------------------------------------------
# Universe
#
# For each commodity:
#   ticker      Yahoo continuous front-month future
#   crop        MapSPAM crop code used as production weight (None for gas)
#   heat_t      threshold (deg C) above which Tmax damages the crop
#               (29 corn, 30 soy: Schlenker and Roberts, 2009)
#   frost_t     threshold (deg C) below which Tmin damages the crop
#   roll        rule used to locate the day Yahoo switches contract (see prices.py)
#   months      delivery months of the futures contract (1 = Jan)
#   zones       production zones; each zone keeps its own growing season
#       iso3    countries kept in the zone
#       adm1    optional list of states / provinces (MapSPAM names)
#       bbox    optional (lat_min, lat_max, lon_min, lon_max) filter
#       box     weather download box that contains the zone
#       season  months where weather is expected to matter (first version,
#               binary; to be refined)
# ---------------------------------------------------------------------------
US_SRW_STATES = ["Illinois", "Indiana", "Ohio", "Kentucky", "Missouri",
                 "Arkansas", "Tennessee", "Michigan", "North Carolina"]
US_HRW_STATES = ["Kansas", "Oklahoma", "Texas", "Colorado", "Nebraska",
                 "South Dakota"]

ALL_MONTHS = list(range(1, 13))

COMMODITIES = {
    "corn": {
        "ticker": "ZC=F", "crop": "maiz", "heat_t": 29.0, "frost_t": 0.0,
        "roll": "cbot", "months": [3, 5, 7, 9, 12],
        "zones": {
            "US": {"iso3": ["USA"], "box": "north_america", "season": [5, 6, 7, 8, 9]},
        },
    },
    "soybeans": {
        "ticker": "ZS=F", "crop": "soyb", "heat_t": 30.0, "frost_t": 0.0,
        "roll": "cbot", "months": [1, 3, 5, 7, 8, 9, 11],
        "zones": {
            "US":  {"iso3": ["USA"], "box": "north_america", "season": [6, 7, 8, 9]},
            "BRA": {"iso3": ["BRA"], "box": "south_america", "season": [11, 12, 1, 2]},
            "ARG": {"iso3": ["ARG"], "box": "south_america", "season": [12, 1, 2, 3]},
        },
    },
    "wheat_srw": {
        "ticker": "ZW=F", "crop": "whea", "heat_t": 32.0, "frost_t": -10.0,
        "roll": "cbot", "months": [3, 5, 7, 9, 12],
        "zones": {
            "US": {"iso3": ["USA"], "adm1": US_SRW_STATES, "box": "north_america",
                   "season": [3, 4, 5, 6]},
            "EU": {"iso3": ["FRA", "DEU", "POL", "ROU"], "box": "europe_black_sea",
                   "season": [3, 4, 5, 6]},
            "BLACK_SEA": {"iso3": ["UKR", "RUS"], "bbox": (42.0, 56.0, 22.0, 50.0),
                          "box": "europe_black_sea", "season": [4, 5, 6, 7]},
        },
    },
    "wheat_hrw": {
        "ticker": "KE=F", "crop": "whea", "heat_t": 32.0, "frost_t": -10.0,
        "roll": "cbot", "months": [3, 5, 7, 9, 12],
        "zones": {
            "US": {"iso3": ["USA"], "adm1": US_HRW_STATES, "box": "north_america",
                   "season": [3, 4, 5, 6]},
        },
    },
    "coffee": {
        "ticker": "KC=F", "crop": "acof", "heat_t": 30.0, "frost_t": 2.0,
        "roll": "ice_coffee", "months": [3, 5, 7, 9, 12],
        "zones": {
            "BRA": {"iso3": ["BRA"], "box": "south_america",
                    "season": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3]},
            "COL": {"iso3": ["COL"], "box": "south_america", "season": ALL_MONTHS},
        },
    },
    "cocoa": {
        "ticker": "CC=F", "crop": "coco", "heat_t": 33.0, "frost_t": 0.0,
        "roll": "ice_cocoa", "months": [3, 5, 7, 9, 12],
        "zones": {
            "WAF": {"iso3": ["CIV", "GHA", "CMR", "NGA"], "box": "west_africa",
                    "season": ALL_MONTHS},
        },
    },
    "sugar": {
        "ticker": "SB=F", "crop": "sugc", "heat_t": 33.0, "frost_t": 2.0,
        "roll": "ice_sugar", "months": [3, 5, 7, 10],
        "zones": {
            "BRA": {"iso3": ["BRA"], "box": "south_america",
                    "season": [10, 11, 12, 1, 2, 3, 6, 7, 8]},
            "IND": {"iso3": ["IND"], "box": "south_asia", "season": [6, 7, 8, 9]},
            "THA": {"iso3": ["THA"], "box": "south_asia", "season": [5, 6, 7, 8, 9, 10]},
        },
    },
    "natgas": {
        "ticker": "NG=F", "crop": None, "heat_t": None, "frost_t": None,
        "roll": "nymex_ng", "months": ALL_MONTHS,
        "zones": {
            # HDD are zero in summer and CDD zero in winter: no season mask needed
            "US": {"iso3": ["USA"], "box": "north_america", "season": ALL_MONTHS},
        },
    },
}

# ---------------------------------------------------------------------------
# Natural gas demand weights: largest US metro areas, 2020 Census population
# (millions). Population-weighted degree days are the industry standard proxy
# for weather-driven gas demand.
# ---------------------------------------------------------------------------
US_METROS = [  # (name, lat, lon, population in millions)
    ("New York", 40.71, -74.01, 20.1), ("Los Angeles", 34.05, -118.24, 13.2),
    ("Chicago", 41.88, -87.63, 9.6), ("Dallas", 32.78, -96.80, 7.6),
    ("Houston", 29.76, -95.37, 7.1), ("Washington", 38.91, -77.04, 6.4),
    ("Philadelphia", 39.95, -75.17, 6.2), ("Miami", 25.76, -80.19, 6.1),
    ("Atlanta", 33.75, -84.39, 6.1), ("Boston", 42.36, -71.06, 4.9),
    ("Phoenix", 33.45, -112.07, 4.8), ("San Francisco", 37.77, -122.42, 4.7),
    ("Riverside", 33.98, -117.38, 4.6), ("Detroit", 42.33, -83.05, 4.4),
    ("Seattle", 47.61, -122.33, 4.0), ("Minneapolis", 44.98, -93.27, 3.7),
    ("San Diego", 32.72, -117.16, 3.3), ("Tampa", 27.95, -82.46, 3.2),
    ("Denver", 39.74, -104.99, 3.0), ("Baltimore", 39.29, -76.61, 2.8),
    ("St. Louis", 38.63, -90.20, 2.8), ("Orlando", 28.54, -81.38, 2.7),
    ("Charlotte", 35.23, -80.84, 2.7), ("San Antonio", 29.42, -98.49, 2.6),
    ("Portland", 45.52, -122.68, 2.5), ("Pittsburgh", 40.44, -80.00, 2.4),
    ("Cincinnati", 39.10, -84.51, 2.3), ("Kansas City", 39.10, -94.58, 2.2),
    ("Columbus", 39.96, -83.00, 2.1), ("Indianapolis", 39.77, -86.16, 2.1),
    ("Cleveland", 41.50, -81.69, 2.1),
]

# Degree-day base used by the US gas market: 65 deg F = 18.3 deg C
DEGREE_DAY_BASE = 18.3
