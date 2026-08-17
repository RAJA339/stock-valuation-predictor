data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  name_prefix = "${var.project}-${var.environment}"

  # Catalog names are SQL identifiers: hyphens are illegal.
  catalog_name = replace("${var.project}_${var.environment}", "-", "_")

  tags = {
    Project     = var.project
    Environment = var.environment
    Owner       = var.owner
    CostCenter  = var.cost_center
    ManagedBy   = "terraform"
  }
}

module "storage" {
  source = "./modules/s3_lakehouse"

  name_prefix   = local.name_prefix
  account_id    = data.aws_caller_identity.current.account_id
  force_destroy = var.force_destroy
  tags          = local.tags
}

module "streaming" {
  source = "./modules/kinesis"

  name_prefix            = local.name_prefix
  stream_name            = var.kinesis_stream_name
  stream_mode            = var.kinesis_stream_mode
  shard_count            = var.kinesis_shard_count
  retention_period_hours = var.kinesis_retention_hours
  alarm_sns_topic_arns   = var.alarm_sns_topic_arns
  tags                   = local.tags
}

module "iam" {
  source = "./modules/iam"

  name_prefix   = local.name_prefix
  account_id    = data.aws_caller_identity.current.account_id
  region        = var.aws_region
  aws_partition = data.aws_partition.current.partition

  databricks_account_id = var.databricks_account_id

  lakehouse_bucket_arns = [
    module.storage.bucket_arns["bronze"],
    module.storage.bucket_arns["silver"],
    module.storage.bucket_arns["gold"],
  ]
  checkpoint_bucket_arn = module.storage.bucket_arns["checkpoints"]
  kinesis_stream_arns   = [module.streaming.stream_arn]
  kms_key_arn           = module.storage.kms_key_arn
  kinesis_kms_key_arn   = module.streaming.kms_key_arn

  tags = local.tags
}

module "unity_catalog" {
  source = "./modules/unity_catalog"

  providers = {
    databricks.account   = databricks.account
    databricks.workspace = databricks.workspace
  }

  name_prefix = local.name_prefix
  environment = var.environment
  region      = var.aws_region

  create_metastore    = var.create_metastore
  metastore_id        = var.metastore_id
  workspace_id        = var.databricks_workspace_id
  bound_workspace_ids = var.bound_workspace_ids

  metastore_owner = var.metastore_owner_group
  catalog_owner   = var.catalog_owner_group

  catalog_name         = local.catalog_name
  catalog_storage_root = module.storage.s3_uris["silver"]

  storage_credential_role_arn = module.iam.unity_catalog_role_arn

  external_locations = module.storage.s3_uris

  schemas = {
    bronze = {
      comment      = "Raw, append-only landing zone. Schema-on-read, no business logic."
      storage_root = module.storage.s3_uris["bronze"]
    }
    silver = {
      comment      = "Cleansed, deduplicated, conformed entities. SCD Type 2 history lives here."
      storage_root = module.storage.s3_uris["silver"]
    }
    gold = {
      comment      = "Business-level aggregates and the vector-search source tables."
      storage_root = module.storage.s3_uris["gold"]
    }
  }

  data_engineer_groups = var.data_engineer_groups
  data_analyst_groups  = var.data_analyst_groups
  force_destroy        = var.force_destroy
}
