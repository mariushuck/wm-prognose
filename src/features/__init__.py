"""Feature builders. Each module exposes:

    build(matches_df, context) -> pandas.DataFrame   # batch, for training
    fixture_features(home, away, neutral, context) -> dict   # one prospective match
    columns() -> list[str]                            # the columns it contributes

A group that is switched off (``feature_groups.yaml``) or whose input data is
absent contributes *no* columns, keeping train/serve feature sets consistent.
"""
