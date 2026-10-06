"""Source normalization; dedup belongs to snapshots."""

from .io import TABLES


def integer(column):
    return (f"CASE WHEN regexp_full_match(trim({column}), '[0-9]+') "
            f"THEN TRY_CAST(trim({column}) AS BIGINT) END")


def calendar_date(column):
    return (f"CASE WHEN regexp_full_match(trim({column}), '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}') "
            f"THEN CAST(try_strptime(trim({column}), '%Y-%m-%d') AS DATE) END")


def boolean(column):
    return f"CASE lower(trim({column})) WHEN 'true' THEN true WHEN 'false' THEN false END"


def clean_file(con, name):
    if name not in TABLES:
        raise ValueError(name)
    if name == "interactions":
        columns = f"""
            {integer('app_id')} AS app_id,
            {integer('helpful')} AS helpful, {integer('funny')} AS funny,
            {calendar_date('date')} AS date,
            {boolean('is_recommended')} AS is_recommended,
            CASE WHEN isfinite(TRY_CAST(hours AS DOUBLE)) AND TRY_CAST(hours AS DOUBLE) >= 0
                 THEN TRY_CAST(hours AS DOUBLE) END AS hours,
            {integer('user_id')} AS user_id,
            {integer('review_id')} AS review_id
        """
        rejection = """CASE WHEN user_id IS NULL OR user_id < 0 THEN 'invalid_user_id'
            WHEN app_id IS NULL OR app_id <= 0 THEN 'invalid_app_id'
            WHEN date IS NULL THEN 'invalid_date'
            WHEN is_recommended IS NULL THEN 'invalid_label'
            WHEN review_id IS NULL OR review_id < 0 THEN 'invalid_review_id' END"""
    elif name == "games":
        columns = f"""
            {integer('app_id')} AS app_id, NULLIF(trim(title), '') AS title,
            {calendar_date('date_release')} AS date_release,
            {boolean('win')} AS win, {boolean('mac')} AS mac, {boolean('linux')} AS linux,
            NULLIF(trim(rating), '') AS rating,
            {integer('positive_ratio')} AS positive_ratio,
            {integer('user_reviews')} AS user_reviews,
            TRY_CAST(price_final AS DOUBLE) AS price_final,
            TRY_CAST(price_original AS DOUBLE) AS price_original,
            TRY_CAST(discount AS DOUBLE) AS discount,
            {boolean('steam_deck')} AS steam_deck
        """
        rejection = "CASE WHEN app_id IS NULL OR app_id <= 0 THEN 'invalid_app_id' END"
    elif name == "users":
        columns = f"""{integer('user_id')} AS user_id,
            {integer('products')} AS products, {integer('reviews')} AS reviews"""
        rejection = "CASE WHEN user_id IS NULL OR user_id < 0 THEN 'invalid_user_id' END"
    else:
        columns = f"""{integer('app_id')} AS app_id,
            coalesce(trim(regexp_replace(description, '\\s+', ' ', 'g')), '') AS description,
            list_sort(list_distinct(list_filter(
                list_transform(coalesce(tags, []::VARCHAR[]), x -> trim(x)),
                x -> x IS NOT NULL AND x <> ''
            ))) AS tags"""
        rejection = "CASE WHEN app_id IS NULL OR app_id <= 0 THEN 'invalid_app_id' END"
    con.execute(f"""CREATE TEMP VIEW {name}_typed AS
        SELECT {columns}, to_json(s) AS raw_record FROM source_{name} s""")
    con.execute(f"""CREATE TEMP VIEW {name}_checked AS
        SELECT *, {rejection} AS rejection_reason FROM {name}_typed""")
    extras = ""
    if name == "content":
        extras = """, description = '' AS missing_description,
            len(tags) = 0 AS missing_tags,
            description = '' AND len(tags) = 0 AS missing_content,
            trim(concat(description, ' ', array_to_string(tags, ' '))) AS text"""
    con.execute(f"""CREATE TEMP VIEW {name}_ready AS
        SELECT * EXCLUDE (raw_record, rejection_reason) {extras}
        FROM {name}_checked WHERE rejection_reason IS NULL""")
    con.execute(f"""CREATE TEMP VIEW {name}_rejected AS
        SELECT * FROM {name}_checked WHERE rejection_reason IS NOT NULL""")
