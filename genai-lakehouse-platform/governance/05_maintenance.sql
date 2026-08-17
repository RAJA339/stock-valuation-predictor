-- ===========================================================================
-- 05 - Table maintenance
--
-- Run hourly from the batch_curation job. Streaming writes produce many small
-- files; without this the read path degrades steadily and the cost curve bends
-- the wrong way about two weeks in.
--
-- Liquid clustering is used instead of partitioning on the query-heavy tables:
-- it adapts to the actual predicate distribution and avoids the small-partition
-- explosion that `PARTITIONED BY (symbol)` would cause on a long tail of
-- rarely-traded instruments.
-- ===========================================================================

USE CATALOG ${catalog};

-- Cluster keys are chosen from how the tables are actually queried:
-- market events by symbol and date, chunks by document, context by symbol.
ALTER TABLE silver.market_events    CLUSTER BY (symbol, event_date);
ALTER TABLE silver.document_chunks  CLUSTER BY (document_id);
ALTER TABLE gold.symbol_daily_metrics CLUSTER BY (symbol, event_date);
ALTER TABLE gold.document_context   CLUSTER BY (symbol, published_date);

OPTIMIZE silver.market_events;
OPTIMIZE silver.document_chunks;
OPTIMIZE silver.instrument_dim;
OPTIMIZE gold.symbol_daily_metrics;
OPTIMIZE gold.document_context;

-- VACUUM below the 7-day default breaks concurrent readers and any time travel
-- an incident investigation might need. 168 hours is the floor, not a target.
VACUUM silver.market_events   RETAIN 168 HOURS;
VACUUM silver.document_chunks RETAIN 168 HOURS;
VACUUM gold.symbol_daily_metrics RETAIN 168 HOURS;

-- Statistics drive join ordering; stale stats on a growing fact table produce
-- a broadcast that should have been a shuffle.
ANALYZE TABLE silver.market_events COMPUTE STATISTICS FOR ALL COLUMNS;
ANALYZE TABLE silver.instrument_dim COMPUTE STATISTICS FOR ALL COLUMNS;
