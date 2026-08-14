# Deployment

## The one setting that matters: `SVP_DATABASE_URL`

Without it, **every track record and watchlist is erased on each redeploy.**

The ledger and the watchlist fall back to a local SQLite file. On Streamlit
Cloud — and on any container host — that filesystem is rebuilt when the app
redeploys, so the file goes with it. The app detects this and says so in the
sidebar and on the Track Record tab, because a record that is silently lost is
worse than one that was never offered: the user believed it was being kept.

Setting one environment variable moves both tables to PostgreSQL and the
problem disappears. No code change, no migration step — the tables are created
on first connection.

### Getting a database

Any PostgreSQL will do. Two with usable free tiers:

- **[Neon](https://neon.tech)** — serverless Postgres, generous free tier,
  connection string is on the project dashboard.
- **[Supabase](https://supabase.com)** — free tier includes Postgres; the
  string is under *Project Settings → Database → Connection string → URI*.

Either gives a URL shaped like:

```
postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

### Setting it on Streamlit Cloud

*Manage app → Settings → Secrets*, then add:

```toml
SVP_DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"
```

Save. The app restarts automatically. The sidebar note under **Track record**
changes from a warning to *"Saved to PostgreSQL — your record persists across
deployments."* That message is the confirmation; there is nothing else to check.

The app reads this from `st.secrets` **and** the environment, so the Secrets UI
above is enough — you do not also need to set an environment variable.
`DATABASE_URL` is accepted as an alias, so a platform that injects that name
automatically (Heroku, Railway, Render) needs no configuration at all.

### Setting it locally

```bash
export SVP_DATABASE_URL="postgresql://localhost/svp"
streamlit run app.py
```

Leave it unset for local development. SQLite is the right choice there — the
file persists between runs on your own machine, which is all local work needs.

### Requirements

`psycopg2-binary` must be installed for the PostgreSQL path to activate. It is
in `requirements.txt`. If the import fails or the connection is refused, the app
falls back to SQLite silently rather than erroring — a database misconfiguration
should not take down the valuation model — so the sidebar note is how you tell
which one is live.

---

## Other environment variables

| Variable | Effect | Default |
|---|---|---|
| `SVP_DATABASE_URL` | PostgreSQL for the cache, ledger and watchlist | unset — SQLite |
| `DATABASE_URL` | Alias for the above | unset |
| `SVP_SQLITE_PATH` | Where the SQLite file lives | `svp_cache.db` |
| `FRED_API_KEY` | Live macro series from FRED | unset — falls back to cached defaults |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Enables the optional synthesis layer over filing quotes | unset — extractive answers |

None are required to run the app. Every one of them degrades to a working
fallback, which is the property that lets the app boot with no configuration at
all.

---

## Python version

Streamlit Cloud currently runs **Python 3.14**. CI covers 3.10 through 3.14, so
a version-specific break should surface in a pull request rather than in
production. The devcontainer pins 3.11 to match the middle of that matrix — no
devcontainers image is published for 3.14, so an exact match with production is
not available locally.
