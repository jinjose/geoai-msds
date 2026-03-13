import pandas as pd

df_2025 = pd.read_parquet("storm_daily_2025.parquet")

# 1️⃣ normalize county
df_2025["county"] = (
    df_2025["county"]
    .str.strip()
    .str.lower()
)

# 2️⃣ convert datetime → date
df_2025["date"] = pd.to_datetime(df_2025["datetime"]).dt.floor("D")

# 3️⃣ derive year
df_2025["year"] = df_2025["date"].dt.year.astype(int)

# 4️⃣ ensure numeric
df_2025["wind_mph"] = pd.to_numeric(df_2025["wind_mph"], errors="coerce")

# 5️⃣ add missing columns
df_2025["event_count"] = 1
df_2025["severe_gust_58"] = (df_2025["wind_mph"] >= 58).astype(int)

# 6️⃣ reorder columns to match old schema
df_2025 = df_2025[
    ["county", "date", "year", "wind_mph", "event_count", "severe_gust_58"]
]

df_2025.to_parquet("storm_daily_2025_standardized.parquet", index=False)