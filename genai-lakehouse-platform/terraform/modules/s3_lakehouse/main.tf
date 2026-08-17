# ---------------------------------------------------------------------------
# S3 lakehouse storage: one bucket per medallion layer plus a dedicated bucket
# for streaming checkpoints / schema registries.
#
# Checkpoints live in their OWN bucket on purpose: they are written at a much
# higher rate than the data itself, they are worthless outside of a running
# stream, and mixing them into a data bucket makes lifecycle rules and Unity
# Catalog external locations awkward to reason about.
# ---------------------------------------------------------------------------

locals {
  # Bucket names are globally unique across all of AWS, so the account id is
  # folded in to keep this module re-usable in more than one account.
  buckets = {
    bronze      = "${var.name_prefix}-bronze-${var.account_id}"
    silver      = "${var.name_prefix}-silver-${var.account_id}"
    gold        = "${var.name_prefix}-gold-${var.account_id}"
    checkpoints = "${var.name_prefix}-checkpoints-${var.account_id}"
  }

  # Raw landing + Bronze keep everything forever-ish; Silver/Gold are
  # reproducible from Bronze, so their noncurrent versions expire sooner.
  noncurrent_expiration_days = {
    bronze      = 365
    silver      = 90
    gold        = 90
    checkpoints = 7
  }
}

resource "aws_kms_key" "lakehouse" {
  description             = "${var.name_prefix} lakehouse SSE-KMS key"
  deletion_window_in_days = var.kms_deletion_window_in_days
  enable_key_rotation     = true
  tags                    = var.tags
}

resource "aws_kms_alias" "lakehouse" {
  name          = "alias/${var.name_prefix}-lakehouse"
  target_key_id = aws_kms_key.lakehouse.key_id
}

resource "aws_s3_bucket" "layer" {
  for_each = local.buckets

  bucket = each.value
  # Guard rails: production buckets must not be destroyable by a stray
  # `terraform destroy`. Dev environments set this to true.
  force_destroy = var.force_destroy

  tags = merge(var.tags, {
    Name           = each.value
    MedallionLayer = each.key
  })
}

resource "aws_s3_bucket_versioning" "layer" {
  for_each = aws_s3_bucket.layer

  bucket = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "layer" {
  for_each = aws_s3_bucket.layer

  bucket = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.lakehouse.arn
    }
    # Without this every GET/PUT is a separate KMS API call; with it S3 caches
    # the data key per bucket/prefix. On a streaming workload this is the
    # difference between a working pipeline and KMS throttling.
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "layer" {
  for_each = aws_s3_bucket.layer

  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "layer" {
  for_each = aws_s3_bucket.layer

  bucket = each.value.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "layer" {
  for_each = aws_s3_bucket.layer

  bucket = each.value.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = local.noncurrent_expiration_days[each.key]
    }
  }

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  # Delta keeps rewriting small files during OPTIMIZE/VACUUM; only the Bronze
  # raw landing zone is cold enough to be worth tiering.
  dynamic "rule" {
    for_each = each.key == "bronze" ? [1] : []
    content {
      id     = "tier-raw-landing-to-ia"
      status = "Enabled"

      filter {
        prefix = "raw/"
      }

      transition {
        days          = 30
        storage_class = "STANDARD_IA"
      }

      transition {
        days          = 180
        storage_class = "GLACIER_IR"
      }
    }
  }
}

# Deny any request that did not arrive over TLS. This is the single most
# commonly failed control in a lakehouse security review.
data "aws_iam_policy_document" "tls_only" {
  for_each = aws_s3_bucket.layer

  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:*"]

    resources = [
      each.value.arn,
      "${each.value.arn}/*",
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "tls_only" {
  for_each = aws_s3_bucket.layer

  bucket = each.value.id
  policy = data.aws_iam_policy_document.tls_only[each.key].json

  # A bucket policy applied before the public access block can briefly trip
  # the "policy would make this bucket public" evaluation.
  depends_on = [aws_s3_bucket_public_access_block.layer]
}

# Unity Catalog external locations expect the well-known medallion prefixes to
# exist so that the "test connection" round-trip has something to list.
resource "aws_s3_object" "layer_prefix" {
  for_each = {
    bronze = "bronze/"
    silver = "silver/"
    gold   = "gold/"
  }

  bucket       = aws_s3_bucket.layer[each.key].id
  key          = each.value
  content_type = "application/x-directory"
  content      = ""
}
