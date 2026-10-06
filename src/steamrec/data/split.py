"""History dedup helper; full snapshot artifacts follow selection of T1/T2."""

from datetime import date


def deduplicate_history(con, cutoff):
    """Return a relation using only reviews observable at cutoff."""
    cutoff = date.fromisoformat(str(cutoff))
    conflict = con.execute("""
        WITH latest AS (
            SELECT *, max(date) OVER (PARTITION BY user_id, app_id) AS latest_date
            FROM interactions WHERE date <= ?
        )
        SELECT user_id, app_id FROM latest WHERE date=latest_date
        GROUP BY user_id, app_id HAVING count(DISTINCT is_recommended) > 1 LIMIT 1
    """, [cutoff]).fetchone()
    if conflict:
        raise ValueError(f"Conflicting labels on latest date for pair {conflict}")
    return con.sql("""
        SELECT * FROM interactions WHERE date <= ?
        QUALIFY row_number() OVER (
            PARTITION BY user_id, app_id ORDER BY date DESC, review_id DESC
        ) = 1
    """, params=[cutoff])
