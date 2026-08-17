project     = "genai-lakehouse"
environment = "prod"
owner       = "data-platform"
cost_center = "CC-1042"

aws_region = "us-east-1"

# Never true in prod. A destroy should fail loudly on a non-empty bucket.
force_destroy = false

kinesis_stream_name     = "market-events"
kinesis_stream_mode     = "PROVISIONED"
kinesis_shard_count     = 8
kinesis_retention_hours = 168

# Metastores are regional and singular: prod attaches to the one dev created.
# Replace with the real id from `terraform -chdir=. output metastore_id` in dev.
create_metastore = false
metastore_id     = "00000000-0000-0000-0000-000000000000"

databricks_account_host   = "https://accounts.cloud.databricks.com"
databricks_account_id     = "00000000-0000-0000-0000-000000000000"
databricks_workspace_host = "https://dbc-11111111-1111.cloud.databricks.com"
databricks_workspace_id   = "6543210987654321"
bound_workspace_ids       = ["6543210987654321"]

metastore_owner_group = "data-platform-admins"
catalog_owner_group   = "data-platform-admins"

# Prod engineers get read/write through CI service principals only; humans read.
data_engineer_groups = ["data-platform-ci"]
data_analyst_groups  = ["data-analysts", "risk-analysts", "genai-app-service"]

alarm_sns_topic_arns = ["arn:aws:sns:us-east-1:000000000000:data-platform-oncall"]
