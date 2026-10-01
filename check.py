import json
import os
import pandas as pd


DATA_DIR = "archive"
REPORT_FILE = "dataset_report.txt"

FILES = {
    "game": "games.csv",
    "recommendation": "recommendations.csv",
    "user": "users.csv",
    "metadata": "games_metadata.json"
}


def write_line(f, text=""):
    f.write(text + "\n")


def format_number(value):
    return f"{value:,}"


def check_csv(path, name, f):
    write_line(f, "=" * 80)
    write_line(f, f"DATASET: {name}")
    write_line(f, f"FILE: {path}")
    write_line(f, "=" * 80)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        write_line(f, f"ERROR: {e}")
        return None

    rows, cols = df.shape

    write_line(f, f"Rows: {format_number(rows)}")
    write_line(f, f"Columns: {cols}")
    write_line(f)

    write_line(f, "COLUMNS")
    write_line(f, "-" * 80)

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique = df[col].nunique(dropna=True)
        null_count = df[col].isna().sum()
        null_rate = null_count / rows * 100 if rows else 0

        write_line(
            f,
            f"{col:<25} "
            f"type={dtype:<12} "
            f"unique={format_number(unique):>10} "
            f"null={format_number(null_count):>10} "
            f"null_rate={null_rate:>7.2f}%"
        )

    write_line(f)
    write_line(f, "DUPLICATES")
    write_line(f, "-" * 80)

    duplicate_rows = df.duplicated().sum()

    write_line(
        f,
        f"Duplicate rows: {format_number(duplicate_rows)} "
        f"({duplicate_rows / rows * 100:.2f}%)"
        if rows
        else "Duplicate rows: 0"
    )

    # Duplicate app_id
    if "app_id" in df.columns:
        duplicate_app_id = df["app_id"].duplicated().sum()

        write_line(
            f,
            f"Duplicate app_id: {format_number(duplicate_app_id)}"
        )

    # Duplicate user_id
    if "user_id" in df.columns:
        duplicate_user_id = df["user_id"].duplicated().sum()

        write_line(
            f,
            f"Duplicate user_id: {format_number(duplicate_user_id)}"
        )

    write_line(f)
    write_line(f, "NUMERIC STATISTICS")
    write_line(f, "-" * 80)

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        write_line(f, "No numeric columns.")
    else:
        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            write_line(
                f,
                f"{col:<25} "
                f"min={series.min()} "
                f"max={series.max()} "
                f"mean={series.mean():.2f} "
                f"median={series.median():.2f}"
            )

    write_line(f)

    # Specific checks
    if name == "game":
        write_line(f, "GAME-SPECIFIC CHECKS")
        write_line(f, "-" * 80)

        for col in [
            "title",
            "date_release",
            "rating",
            "positive_ratio",
            "user_reviews",
            "price_final",
            "price_original",
            "discount"
        ]:
            if col in df.columns:
                empty = df[col].isna().sum()

                if df[col].dtype == "object":
                    empty += (df[col].astype(str).str.strip() == "").sum()

                write_line(
                    f,
                    f"{col:<25} missing/empty={format_number(empty)}"
                )

        write_line(f)

    if name == "recommendation":
        write_line(f, "RECOMMENDATION-SPECIFIC CHECKS")
        write_line(f, "-" * 80)

        if "is_recommended" in df.columns:
            write_line(f, "is_recommended distribution:")
            distribution = df["is_recommended"].value_counts(dropna=False)

            for value, count in distribution.items():
                write_line(
                    f,
                    f"  {value}: {format_number(count)}"
                )

        if "hours" in df.columns:
            zero_hours = (df["hours"] == 0).sum()
            negative_hours = (df["hours"] < 0).sum()

            write_line(
                f,
                f"hours = 0: {format_number(zero_hours)}"
            )

            write_line(
                f,
                f"hours < 0: {format_number(negative_hours)}"
            )

        write_line(f)

    if name == "user":
        write_line(f, "USER-SPECIFIC CHECKS")
        write_line(f, "-" * 80)

        if "user_id" in df.columns:
            unique_users = df["user_id"].nunique()

            write_line(
                f,
                f"Unique users: {format_number(unique_users)}"
            )

        if "products" in df.columns:
            empty_products = (
                df["products"].isna() |
                (df["products"].astype(str).str.strip() == "")
            ).sum()

            write_line(
                f,
                f"Empty products: {format_number(empty_products)}"
            )

        if "reviews" in df.columns:
            empty_reviews = (
                df["reviews"].isna() |
                (df["reviews"].astype(str).str.strip() == "")
            ).sum()

            write_line(
                f,
                f"Empty reviews: {format_number(empty_reviews)}"
            )

        write_line(f)

    return df


def check_metadata(path, f):
    write_line(f, "=" * 80)
    write_line(f, "DATASET: metadata")
    write_line(f, f"FILE: {path}")
    write_line(f, "=" * 80)

    total = 0

    missing_description = 0
    missing_tags = 0
    missing_both = 0

    duplicate_app_id = 0

    app_ids = set()

    try:
        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    write_line(
                        f,
                        f"Invalid JSON at line {line_number}: {e}"
                    )
                    continue

                total += 1

                app_id = row.get("app_id")

                if app_id in app_ids:
                    duplicate_app_id += 1

                app_ids.add(app_id)

                description = row.get("description")
                tags = row.get("tags")

                no_description = not description
                no_tags = not tags

                if no_description:
                    missing_description += 1

                if no_tags:
                    missing_tags += 1

                if no_description and no_tags:
                    missing_both += 1

    except Exception as e:
        write_line(f, f"ERROR: {e}")
        return None

    write_line(f, f"Rows: {format_number(total)}")
    write_line(f, f"Columns: app_id, description, tags")
    write_line(f)

    write_line(f, "MISSING DATA")
    write_line(f, "-" * 80)

    write_line(
        f,
        f"Description empty/null: "
        f"{format_number(missing_description)} "
        f"({missing_description / total * 100:.2f}%)"
        if total
        else "Description empty/null: 0"
    )

    write_line(
        f,
        f"Tags empty/null: "
        f"{format_number(missing_tags)} "
        f"({missing_tags / total * 100:.2f}%)"
        if total
        else "Tags empty/null: 0"
    )

    write_line(
        f,
        f"Both empty/null: "
        f"{format_number(missing_both)} "
        f"({missing_both / total * 100:.2f}%)"
        if total
        else "Both empty/null: 0"
    )

    write_line(f)
    write_line(f, "DUPLICATES")
    write_line(f, "-" * 80)

    write_line(
        f,
        f"Duplicate app_id: {format_number(duplicate_app_id)}"
    )

    write_line(f)

    return app_ids


def main():
    report_path = os.path.join(DATA_DIR, REPORT_FILE)

    game_df = None
    recommendation_df = None
    user_df = None
    metadata_app_ids = None

    with open(report_path, "w", encoding="utf-8") as f:

        write_line(f, "STEAM DATASET QUALITY REPORT")
        write_line(f, "=" * 80)
        write_line(f, f"Data directory: {DATA_DIR}")
        write_line(f)

        # Game
        path = os.path.join(DATA_DIR, FILES["game"])

        if os.path.exists(path):
            game_df = check_csv(path, "game", f)
        else:
            write_line(f, f"Missing file: {path}")

        # Recommendation
        path = os.path.join(DATA_DIR, FILES["recommendation"])

        if os.path.exists(path):
            recommendation_df = check_csv(
                path,
                "recommendation",
                f
            )
        else:
            write_line(f, f"Missing file: {path}")

        # User
        path = os.path.join(DATA_DIR, FILES["user"])

        if os.path.exists(path):
            user_df = check_csv(path, "user", f)
        else:
            write_line(f, f"Missing file: {path}")

        # Metadata
        path = os.path.join(DATA_DIR, FILES["metadata"])

        if os.path.exists(path):
            metadata_app_ids = check_metadata(path, f)
        else:
            write_line(f, f"Missing file: {path}")

        # Cross-dataset checks
        write_line(f, "=" * 80)
        write_line(f, "CROSS-DATASET CHECKS")
        write_line(f, "=" * 80)

        if game_df is not None and "app_id" in game_df.columns:

            game_app_ids = set(
                game_df["app_id"].dropna()
            )

            write_line(
                f,
                f"Game unique app_id: "
                f"{format_number(len(game_app_ids))}"
            )

            if metadata_app_ids is not None:

                intersection = (
                    game_app_ids &
                    metadata_app_ids
                )

                game_only = (
                    game_app_ids -
                    metadata_app_ids
                )

                metadata_only = (
                    metadata_app_ids -
                    game_app_ids
                )

                write_line(f)
                write_line(f, "GAME ↔ METADATA")
                write_line(f, "-" * 80)

                write_line(
                    f,
                    f"Matching app_id: "
                    f"{format_number(len(intersection))}"
                )

                write_line(
                    f,
                    f"Game only: "
                    f"{format_number(len(game_only))}"
                )

                write_line(
                    f,
                    f"Metadata only: "
                    f"{format_number(len(metadata_only))}"
                )

                if game_app_ids:
                    coverage = (
                        len(intersection) /
                        len(game_app_ids) *
                        100
                    )

                    write_line(
                        f,
                        f"Metadata coverage of game: "
                        f"{coverage:.2f}%"
                    )

        if recommendation_df is not None:

            if "app_id" in recommendation_df.columns:
                recommendation_app_ids = set(
                    recommendation_df["app_id"].dropna()
                )

                write_line(f)
                write_line(f, "GAME ↔ RECOMMENDATION")
                write_line(f, "-" * 80)

                write_line(
                    f,
                    f"Unique recommendation app_id: "
                    f"{format_number(len(recommendation_app_ids))}"
                )

                if game_df is not None:
                    game_app_ids = set(
                        game_df["app_id"].dropna()
                    )

                    matched = (
                        game_app_ids &
                        recommendation_app_ids
                    )

                    write_line(
                        f,
                        f"Matching app_id: "
                        f"{format_number(len(matched))}"
                    )

                    write_line(
                        f,
                        f"Recommendation app_id not in game: "
                        f"{format_number(len(recommendation_app_ids - game_app_ids))}"
                    )

        if recommendation_df is not None:

            if "user_id" in recommendation_df.columns:
                recommendation_users = set(
                    recommendation_df["user_id"].dropna()
                )

                write_line(f)
                write_line(f, "USER ↔ RECOMMENDATION")
                write_line(f, "-" * 80)

                write_line(
                    f,
                    f"Unique recommendation users: "
                    f"{format_number(len(recommendation_users))}"
                )

                if user_df is not None:
                    user_users = set(
                        user_df["user_id"].dropna()
                    )

                    matched = (
                        user_users &
                        recommendation_users
                    )

                    write_line(
                        f,
                        f"Matching user_id: "
                        f"{format_number(len(matched))}"
                    )

                    write_line(
                        f,
                        f"Recommendation users not in user: "
                        f"{format_number(len(recommendation_users - user_users))}"
                    )

        write_line(f)
        write_line(f, "REPORT FINISHED")
        write_line(f, "=" * 80)

    print(f"Done!")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()