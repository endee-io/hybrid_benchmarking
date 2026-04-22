from typing import Dict

from src.interface import HybridDB
from src.dbs.endee.db import EndeeDB
from src.dbs.qdrant.db import QdrantDB
from src.dbs.vespa.db import VespaDB

DB_REGISTRY: Dict[str, type] = {
    "endee":  EndeeDB,
    "qdrant": QdrantDB,
    "vespa":  VespaDB,
}


def create_db(db_name: str, db_config: dict) -> HybridDB:
    cls = DB_REGISTRY.get(db_name)
    if cls is None:
        raise ValueError(f"Unknown DB: '{db_name}'. Available: {list(DB_REGISTRY)}")
    return cls(**db_config)


def add_all_db_args(parser) -> None:
    """Register argparse arguments for every DB in the registry."""
    for cls in DB_REGISTRY.values():
        cls.add_args(parser)


def build_db_config(db_name: str, args) -> dict:
    """Build the DB-specific config dict from parsed args."""
    cls = DB_REGISTRY.get(db_name)
    if cls is None:
        raise ValueError(f"Unknown DB: '{db_name}'. Available: {list(DB_REGISTRY)}")
    return cls.build_config(args)
