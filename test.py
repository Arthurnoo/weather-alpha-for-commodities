import pandas as pd

components = pd.read_parquet("data/processed/components_weekly.parquet")
wsi = pd.read_parquet("data/processed/wsi_cpri_weekly.parquet")

components.to_csv("components_weekly.csv")
wsi.to_csv("wsi_cpri_weekly.csv")