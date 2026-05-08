import logging
import time
import uuid
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Datatype,
    Distance,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    Modifier,
    OptimizersConfigDiff,
    PointStruct,
    Prefetch,
    SearchParams,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.interface import HybridDB

logger = logging.getLogger(__name__)

MAX_RETRIES = 10

_DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "dot":    Distance.DOT,
    "l2":     Distance.EUCLID,
}


def _to_qdrant_id(raw_id: str):
    """Convert a string ID to a Qdrant-compatible int or UUID string."""
    try:
        return int(raw_id)
    except (ValueError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))


class QdrantDB(HybridDB):
    """Qdrant implementation of HybridDB."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        sparse_vector_name: str = "sparse",
        dense_vector_name: str = "dense",
        query_mode: str = "hybrid",
        modifier: str = "none",
        on_disk_index: bool = False,
        segment_number: int = 8,
        datatype: str = "float32",
        ef_construction: int = 128,
        ef_search: int = 128,
        hnsw_m: int = 16,
    ):
        self.client             = QdrantClient(host=host, port=port, prefer_grpc=False)
        self.sparse_vector_name = sparse_vector_name
        self.dense_vector_name  = dense_vector_name
        self.query_mode         = query_mode
        self.modifier           = Modifier.IDF if modifier.lower() == "idf" else Modifier.NONE
        self.on_disk_index      = on_disk_index
        self.segment_number     = segment_number
        self.datatype           = {"float32": Datatype.FLOAT32, "float16": Datatype.FLOAT16, "uint8": Datatype.UINT8}.get(datatype.lower(), Datatype.FLOAT32)
        self.ef_construction    = ef_construction
        self.ef_search          = ef_search
        self.hnsw_m             = hnsw_m
        self.collection         = None
        logger.info("QdrantDB connected to %s:%d (datatype=%s, ef_construction=%d, ef_search=%d)", host, port, datatype, ef_construction, ef_search)

    def init(
        self,
        index_name: str,
        dimension: int = 384,
        space_type: str = "cosine",
        create: bool = True,
        **kwargs,
    ) -> None:
        try:
            self.collection = index_name
            if create:
                distance = _DISTANCE_MAP.get(space_type.lower(), Distance.COSINE)
                existing = {c.name for c in self.client.get_collections().collections}
                if index_name in existing:
                    logger.info("Collection '%s' already exists — skipping creation", index_name)
                else:
                    self.client.create_collection(
                        collection_name=index_name,
                        vectors_config={
                            self.dense_vector_name: VectorParams(
                                size=dimension,
                                distance=distance,
                                datatype=self.datatype,
                                on_disk=self.on_disk_index,
                                hnsw_config=HnswConfigDiff(ef_construct=self.ef_construction, m=self.hnsw_m, on_disk=self.on_disk_index),
                            )
                        },
                        sparse_vectors_config={
                            self.sparse_vector_name: SparseVectorParams(
                                index=SparseIndexParams(on_disk=self.on_disk_index),
                                modifier=self.modifier,
                            )
                        },
                        optimizers_config=OptimizersConfigDiff(
                            default_segment_number=self.segment_number,
                        ),
                    )
                    logger.info(
                        "Created collection '%s' (dim=%d, space=%s, modifier=%s)",
                        index_name, dimension, space_type, self.modifier,
                    )
            logger.info("Connected to collection '%s'", index_name)
        except Exception as e:
            logger.error("init failed for collection '%s': %s", index_name, e)
            raise

    def index_batch(self, points: List[Dict]) -> None:
        try:
            qdrant_points = []
            for p in points:
                if not p["sparse_indices"]:
                    continue
                vectors = {
                    self.sparse_vector_name: SparseVector(
                        indices=p["sparse_indices"],
                        values=p["sparse_values"],
                    ),
                    self.dense_vector_name: p["vector"],
                }
                payload = {"original_id": str(p["id"])}
                payload.update(p.get("meta", {}))
                qdrant_points.append(PointStruct(
                    id=_to_qdrant_id(str(p["id"])),
                    vector=vectors,
                    payload=payload,
                ))

            if not qdrant_points:
                return

            for attempt in range(MAX_RETRIES):
                try:
                    self.client.upsert(
                        collection_name=self.collection,
                        points=qdrant_points,
                        wait=True,
                    )
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

    def wait_for_index(self) -> None:
        """Poll until Qdrant reports CollectionStatus.GREEN (HNSW graph fully built)."""
        logger.info("Waiting for Qdrant collection '%s' to reach GREEN status...", self.collection)
        while True:
            info = self.client.get_collection(self.collection)
            if info.status == CollectionStatus.GREEN:
                logger.info("Collection '%s' is GREEN — index fully built", self.collection)
                return
            logger.debug("Collection status: %s — waiting...", info.status)
            time.sleep(5)

    def search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
        text: str = "",
    ) -> List[Dict]:
        try:
            if self.query_mode == "hybrid":
                results = self.client.query_points(
                    collection_name=self.collection,
                    prefetch=[
                        Prefetch(
                            query=dense_vector,
                            using=self.dense_vector_name,
                            limit=top_k,
                            params=SearchParams(hnsw_ef=self.ef_search),
                        ),
                        Prefetch(
                            query=SparseVector(indices=sparse_indices, values=sparse_values),
                            using=self.sparse_vector_name,
                            limit=top_k,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                ).points
            else:
                results = self.client.query_points(
                    collection_name=self.collection,
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using=self.sparse_vector_name,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                ).points
            return [{"id": str(r.payload.get("original_id")), "score": r.score} for r in results]
        except Exception as e:
            logger.error("search failed: %s", e)
            raise

    def list_indices(self) -> List[str]:
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            logger.error("list_indices failed: %s", e)
            raise

    @staticmethod
    def add_args(parser) -> None:
        g = parser.add_argument_group("Qdrant options")
        g.add_argument("--host",               default="localhost",
                       help="[Qdrant] Host (default: localhost)")
        g.add_argument("--port",               type=int, default=6333,
                       help="[Qdrant] Port (default: 6333)")
        g.add_argument("--sparse-vector-name", default="sparse",
                       help="[Qdrant] Sparse vector field name (default: sparse)")
        g.add_argument("--dense-vector-name",  default="dense",
                       help="[Qdrant] Dense vector field name (default: dense)")
        g.add_argument("--query-mode",         default="hybrid", choices=["hybrid", "sparse"],
                       help="[Qdrant] Query mode: hybrid (RRF) or sparse (default: hybrid)")
        g.add_argument("--modifier",           default="none", choices=["none", "idf"],
                       help="[Qdrant] Sparse vector modifier (default: none)")
        g.add_argument("--on-disk-index",      action="store_true", default=False,
                       help="[Qdrant] Store vectors, HNSW graph, and sparse index on disk (default: False, in-memory)")
        g.add_argument("--segment-number",     type=int, default=8,
                       help="[Qdrant] Number of collection segments (default: 8)")
        g.add_argument("--datatype",           default="float32", choices=["float32", "float16", "uint8"],
                       help="[Qdrant] Vector storage datatype (default: float32)")
        g.add_argument("--ef-construction",    type=int, default=128,
                       help="[Qdrant] HNSW ef_construct during indexing (default: 128)")
        g.add_argument("--ef-search",          type=int, default=128,
                       help="[Qdrant] HNSW ef during search (default: 128)")
        g.add_argument("--hnsw-m",             type=int, default=16,
                       help="[Qdrant] HNSW m — links per node (default: 16)")

    @staticmethod
    def build_config(args) -> dict:
        return {
            "host":               args.host,
            "port":               args.port,
            "sparse_vector_name": args.sparse_vector_name,
            "dense_vector_name":  args.dense_vector_name,
            "query_mode":         args.query_mode,
            "modifier":           args.modifier,
            "on_disk_index":      args.on_disk_index,
            "segment_number":     args.segment_number,
            "datatype":           args.datatype,
            "ef_construction":    args.ef_construction,
            "ef_search":          args.ef_search,
            "hnsw_m":             args.hnsw_m,
        }
