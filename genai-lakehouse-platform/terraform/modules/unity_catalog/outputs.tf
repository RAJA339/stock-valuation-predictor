output "metastore_id" {
  description = "Id of the metastore backing this environment."
  value       = local.metastore_id
}

output "catalog_name" {
  description = "Unity Catalog catalog name."
  value       = databricks_catalog.this.name
}

output "schema_names" {
  description = "Fully qualified medallion schema names."
  value       = { for k, s in databricks_schema.medallion : k => "${databricks_catalog.this.name}.${s.name}" }
}

output "storage_credential_name" {
  description = "Name of the Unity Catalog storage credential."
  value       = databricks_storage_credential.this.name
}

output "external_location_names" {
  description = "Map of layer -> external location name."
  value       = { for k, l in databricks_external_location.layer : k => l.name }
}

output "landing_volume_path" {
  description = "UC volume path Auto Loader reads unstructured documents from."
  value       = "/Volumes/${databricks_catalog.this.name}/${databricks_schema.medallion["bronze"].name}/${databricks_volume.landing.name}"
}

output "checkpoint_volume_path" {
  description = "UC volume path used for Structured Streaming checkpoints."
  value       = "/Volumes/${databricks_catalog.this.name}/${databricks_schema.medallion["bronze"].name}/${databricks_volume.checkpoints.name}"
}
