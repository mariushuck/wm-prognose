"""Data-source loaders. Each module exposes:

    load(cfg: dict, force: bool = False) -> pandas.DataFrame

returning a DataFrame with that source's documented, fixed column set. A
disabled or not-yet-implemented source returns an *empty* DataFrame with those
columns, so the ingestion pipeline runs regardless of which layers are active.
"""
