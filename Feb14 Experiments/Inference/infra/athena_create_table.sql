CREATE EXTERNAL TABLE IF NOT EXISTS geoai_predictions (
  prediction double
)
PARTITIONED BY (
  run_date string,
  cutoff_stage string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
LOCATION 's3://YOUR_BUCKET/predictions/';

-- After data lands:
-- MSCK REPAIR TABLE geoai_predictions;
