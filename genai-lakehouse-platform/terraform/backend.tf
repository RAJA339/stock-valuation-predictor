# Remote state. Values are supplied per environment via `-backend-config`:
#
#   terraform init -backend-config=envs/dev/backend.hcl
#
# The state bucket and lock table are intentionally NOT managed here -- a
# bootstrap that manages its own backend cannot be destroyed or recovered
# cleanly. Create them once, out of band (see README, "Bootstrap").
terraform {
  backend "s3" {
    key          = "genai-lakehouse/terraform.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}
