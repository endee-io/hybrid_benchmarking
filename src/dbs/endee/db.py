import logging
import time
from typing import Dict, List

from endee import Endee, rerank

from src.interface import HybridDB

logger = logging.getLogger(__name__)

DEV_PATH = "https://dev.endee.io/api/v2"
MAX_RETRIES = 10

DENSE_FIELD = "embedding"
SPARSE_FIELD = "keywords"


class EndeeDB(HybridDB):
    """Endee implementation of HybridDB (v2 Collections API)."""

    def __init__(
        self,
        vector_token: str,
        base_url: str = DEV_PATH,
        sparse_scoring_model: str = "default",
        precision: str = "float32",
        query_mode: str = "hybrid",
        m: int = 16,
        ef_con: int = 128,
        ef_search: int = 128,
    ):
        self.vx = Endee(token=vector_token)
        self.vx.set_base_url(base_url)
        self.sparse_scoring_model = sparse_scoring_model
        self.precision = precision
        self.query_mode = query_mode
        self.m = m
        self.ef_con = ef_con
        self.ef_search = ef_search
        self.collection = None
        logger.info("EndeeDB connected to %s (query_mode=%s, M=%d, ef_con=%d, ef_search=%d)",
                    base_url, query_mode, m, ef_con, ef_search)

    def init(
        self,
        index_name: str,
        dimension: int = 0,
        space_type: str = "cosine",
        create: bool = True,
        **kwargs,
    ) -> None:
        try:
            if create:
                fields = [
                    {
                        "name": DENSE_FIELD,
                        "type": "vector",
                        "params": {
                            "dimension": dimension,
                            "space_type": space_type,
                            "precision": self.precision,
                            "M": self.m,
                            "ef_con": self.ef_con,
                        },
                    },
                    {
                        "name": SPARSE_FIELD,
                        "type": "sparse",
                        "sparse_model": self.sparse_scoring_model,
                    },
                ]
                self.vx.create_collection(name=index_name, fields=fields)
                logger.info("Created collection '%s' (dim=%d, space=%s)", index_name, dimension, space_type)
            self.collection = self.vx.get_collection(index_name)
            logger.info("Connected to collection '%s'", index_name)
        except Exception as e:
            logger.error("init failed for collection '%s': %s", index_name, e)
            raise

    def index_batch(self, points: List[Dict]) -> None:
        objects = [
            {
                "id": p["id"],
                "meta": p.get("meta", {}),
                "fields": {
                    DENSE_FIELD: p["vector"],
                    SPARSE_FIELD: {
                        "indices": p["sparse_indices"],
                        "values":  p["sparse_values"],
                    },
                },
            }
            for p in points
        ]
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    self.collection.upsert(objects)
                    return
                except Exception as e:
                    logger.warning("index_batch attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1.5)
                    else:
                        raise
        except Exception as e:
            logger.error("index_batch failed after %d retries (%d points): %s", MAX_RETRIES, len(points), e)
            raise

    def search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
        text: str = "",
    ) -> List[Dict]:
        try:
            sparse_query = {"indices": sparse_indices, "values": sparse_values}
            if self.query_mode == "sparse":
                fields = {SPARSE_FIELD: {"query": sparse_query, "limit": top_k}}
                raw = self.collection.search(fields=fields, ef_search=self.ef_search)
                hits = raw["results"][SPARSE_FIELD]
            else:
                fields = {
                    DENSE_FIELD:  {"query": dense_vector,  "limit": top_k},
                    SPARSE_FIELD: {"query": sparse_query,  "limit": top_k},
                }
                raw = self.collection.search(fields=fields, ef_search=self.ef_search)
                hits = rerank(raw, name="rrf", limit=top_k)["results"]
            return [{"id": str(p["id"]), "score": p["similarity"]} for p in hits]
        except Exception as e:
            logger.error("search failed: %s", e)
            raise

    def list_indices(self) -> List[str]:
        try:
            collections = self.vx.list_collections()
            return [c["name"] for c in collections if isinstance(c, dict)]
        except Exception as e:
            logger.error("list_indices failed: %s", e)
            raise

    @staticmethod
    def add_args(parser) -> None:
        g = parser.add_argument_group("Endee options")
        g.add_argument("--vector-token",         default="12345678",
                       help="[Endee] API token (default: 12345678)")
        g.add_argument("--base-url",             default=DEV_PATH,
                       help=f"[Endee] Base URL (default: {DEV_PATH})")
        g.add_argument("--sparse-scoring-model", default="default",
                       help="[Endee] Sparse scoring model (default: default) or for bm25 use endee_bm25")
        g.add_argument("--precision",            default="float32",
                       help="[Endee] Vector precision (default: float32)")
        g.add_argument("--query-mode",           default="hybrid", choices=["hybrid", "sparse"],
                       help="[Endee] Query mode: hybrid (dense+sparse) or sparse only (default: hybrid)")
        g.add_argument("--m",                    type=int, default=16,
                       help="[Endee] HNSW M — number of bi-directional links per node (default: 16)")
        g.add_argument("--ef-con",               type=int, default=128,
                       help="[Endee] HNSW ef_construction — candidates during index build (default: 128)")
        g.add_argument("--ef-search",            type=int, default=128,
                       help="[Endee] HNSW ef_search — candidates during query (default: 128)")

    @staticmethod
    def build_config(args) -> dict:
        return {
            "vector_token":         args.vector_token,
            "base_url":             args.base_url,
            "sparse_scoring_model": args.sparse_scoring_model,
            "precision":            args.precision,
            "query_mode":           args.query_mode,
            "m":                    args.m,
            "ef_con":               args.ef_con,
            "ef_search":            args.ef_search,
        }
