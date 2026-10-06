"""Validate staging artifacts before persisting clean tables."""

from .io import TABLES, sql_path

REQUIRED = {
    "interactions": ("app_id", "helpful", "funny", "date", "is_recommended", "hours", "user_id", "review_id"),
    "games": ("app_id", "title", "date_release", "win", "mac", "linux", "rating", "positive_ratio",
              "user_reviews", "price_final", "price_original", "discount", "steam_deck"),
    "users": ("user_id", "products", "reviews"),
    "content": {"app_id", "description", "tags"},
}


def validate_schema(con, name):
    columns = {row[0] for row in con.execute(f"DESCRIBE source_{name}").fetchall()}
    missing = set(REQUIRED[name]) - columns
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    extra = columns - set(REQUIRED[name])
    if extra:
        raise ValueError(f"{name}: unexpected columns {sorted(extra)}; refusing to discard columns")


def validate_staged(con, staged, rejected):
    counts = {}
    for name in TABLES:
        con.execute(f"CREATE TEMP VIEW checked_{name} AS SELECT * FROM read_parquet({sql_path(staged[name])})")
        actual_columns = [row[0] for row in con.execute(f"DESCRIBE checked_{name}").fetchall()]
        if name != "content" and actual_columns != list(REQUIRED[name]):
            raise RuntimeError(f"{name}: output columns/order differ from required schema")
        source_rows = con.execute(f"SELECT count(*) FROM source_{name}").fetchone()[0]
        clean_rows = con.execute(f"SELECT count(*) FROM checked_{name}").fetchone()[0]
        rejected_rows = con.execute(f"SELECT count(*) FROM read_parquet({sql_path(rejected[name])})").fetchone()[0]
        if source_rows != clean_rows + rejected_rows:
            raise RuntimeError(f"{name}: row accounting failed")
        counts[name] = {"input_rows": source_rows, "stored_rows": clean_rows, "rejected_rows": rejected_rows}
        if name != "interactions":
            key = "user_id" if name == "users" else "app_id"
            duplicate = con.execute(f"""SELECT {key}, count(*) FROM checked_{name}
                GROUP BY {key} HAVING count(*) > 1 LIMIT 1""").fetchone()
            if duplicate:
                raise ValueError(f"{name}: duplicate normalized key {duplicate}; no automatic dedup")
    duplicate_review = con.execute("""SELECT review_id FROM checked_interactions
        GROUP BY review_id HAVING count(*) > 1 LIMIT 1""").fetchone()
    if duplicate_review:
        raise ValueError(f"Duplicate review_id: {duplicate_review[0]}")

    def number(query):
        return con.execute(query).fetchone()[0]

    audit = {
        "positive_rows": number("SELECT count(*) FROM checked_interactions WHERE is_recommended"),
        "negative_rows": number("SELECT count(*) FROM checked_interactions WHERE NOT is_recommended"),
        "repeated_pairs": number("""SELECT count(*) FROM (
            SELECT user_id, app_id FROM checked_interactions GROUP BY user_id, app_id HAVING count(*) > 1)"""),
        "orphan_game_reviews": number("""SELECT count(*) FROM checked_interactions i
            WHERE NOT EXISTS (SELECT 1 FROM checked_games g WHERE g.app_id=i.app_id)"""),
        "orphan_user_reviews": number("""SELECT count(*) FROM checked_interactions i
            WHERE NOT EXISTS (SELECT 1 FROM checked_users u WHERE u.user_id=i.user_id)"""),
        "reviews_before_release": number("""SELECT count(*) FROM checked_interactions i
            JOIN checked_games g USING (app_id) WHERE i.date < g.date_release"""),
        "missing_release_dates": number("SELECT count(*) FROM checked_games WHERE date_release IS NULL"),
        "missing_content": number("SELECT count(*) FROM checked_content WHERE missing_content"),
        "games_without_metadata": number("""SELECT count(*) FROM checked_games g
            WHERE NOT EXISTS (SELECT 1 FROM checked_content c WHERE c.app_id=g.app_id)"""),
        "metadata_without_game": number("""SELECT count(*) FROM checked_content c
            WHERE NOT EXISTS (SELECT 1 FROM checked_games g WHERE g.app_id=c.app_id)"""),
    }
    months = con.execute("""SELECT strftime(date, '%Y-%m') AS month, count(*) AS rows,
        count(*) FILTER (WHERE is_recommended) AS positives,
        count(DISTINCT user_id) AS users FROM checked_interactions
        GROUP BY month ORDER BY month""").fetchall()
    return counts, audit, [dict(zip(("month", "rows", "positives", "users"), row)) for row in months]
