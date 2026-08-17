project     = "genai-lakehouse"
environment = "dev"
owner       = "data-platform"
cost_center = "CC-1042"

aws_region = "us-east-1"

# Dev is disposable on purpose -- `terraform destroy` must actually work.
force_destroy = true

kinesis_stream_name     = "market-events"
kinesis_stream_mode     = "ON_DEMAND"
kinesis_retention_hours = 24

# The dev environment owns the regional metastore; staging and prod attach to
# it by id. See envs/prod/terraform.tfvars.
create_metastore = true
metastore_id     = ""

databricks_account_host   = "https://accounts.cloud.databricks.com"
databricks_account_id     = "00000000-0000-0000-0000-000000000000"
databricks_workspace_host = "https://dbc-00000000-0000.cloud.databricks.com"
databricks_workspace_id   = "1234567890123456"
bound_workspace_ids       = ["1234567890123456"]

metastore_owner_group = "data-platform-admins"
catalog_owner_group   = "data-platform-admins"
data_engineer_groups  = ["data-engineers"]
data_analyst_groups   = ["data-analysts", "genai-app-service"]

# databricks_client_id / databricks_client_secret come from the environment:
#   export TF_VAR_databricks_client_id=...
#   export TF_VAR_databricks_client_secret=...
