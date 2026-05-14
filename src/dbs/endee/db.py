import logging
import time
from typing import Dict, List

from endee import Endee

from src.interface import HybridDB

logger = logging.getLogger(__name__)

DEV_PATH = "https://dev.endee.io/api/v1"
MAX_RETRIES = 10


class EndeeDB(HybridDB):
    """Endee implementation of HybridDB."""

    def __init__(
        self,
        vector_token: str,
        base_url: str = DEV_PATH,
        sparse_scoring_model: str = "default",
        precision: str = "float32",
        query_mode: str = "hybrid",
    ):
        self.vx = Endee(token=vector_token)
        self.vx.set_base_url(base_url)
        self.sparse_scoring_model = sparse_scoring_model
        self.precision = precision
        self.query_mode = query_mode
        self.index = None
        logger.info("EndeeDB connected to %s (query_mode=%s)", base_url, query_mode)

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
                self.vx.create_index(
                    name=index_name,
                    dimension=dimension,
                    space_type=space_type,
                    sparse_model=self.sparse_scoring_model,
                    precision=self.precision,
                )
                logger.info("Created index '%s' (dim=%d, space=%s)", index_name, dimension, space_type)
            self.index = self.vx.get_index(index_name)
            logger.info("Connected to index '%s'", index_name)
        except Exception as e:
            logger.error("init failed for index '%s': %s", index_name, e)
            raise

    def index_batch(self, points: List[Dict]) -> None:
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    self.index.upsert(points)
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
            if self.query_mode == "sparse":
                raw = self.index.query(
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                    top_k=top_k,
                )
            else:
                raw = self.index.query(
                    vector=dense_vector,
                    sparse_indices=sparse_indices,
                    sparse_values=sparse_values,
                    top_k=top_k,
                )
            return [{"id": str(p["meta"]["id"]), "score": p["similarity"]} for p in raw]
        except Exception as e:
            logger.error("search failed: %s", e)
            raise

    def list_indices(self) -> List[str]:
        try:
            return self.vx.list_indices()
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

    @staticmethod
    def build_config(args) -> dict:
        return {
            "vector_token":         args.vector_token,
            "base_url":             args.base_url,
            "sparse_scoring_model": args.sparse_scoring_model,
            "precision":            args.precision,
            "query_mode":           args.query_mode,
        }
