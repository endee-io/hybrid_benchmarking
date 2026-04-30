import logging
import time
from typing import Dict, List

from pymilvus import (
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker,
)

from src.interface import HybridDB

logger = logging.getLogger(__name__)

MAX_RETRIES = 10

_METRIC_MAP = {
    "cosine": "COSINE",
    "dot":    "IP",
    "l2":     "L2",
}

# Sparse modes that use pre-computed (indices, values) pairs uploaded as SPARSE_FLOAT_VECTOR.
# builtin_bm25 is the exception: Milvus computes sparse vectors from raw text server-side.
_PRECOMPUTED_MODES = {"bm25", "pymilvus_bm25", "splade", "milvus_splade"}


def _dense_search_params(index_type: str, nprobe: int, ef_search: int) -> dict:
    if index_type in ("IVF_FLAT", "IVF_SQ8"):
        return {"nprobe": nprobe}
    if index_type == "HNSW":
        return {"ef": ef_search}
    return {}


class MilvusDB(HybridDB):
    """
    Milvus implementation of HybridDB supporting 5 sparse modes:

      bm25           – pre-computed rank_bm25 sparse vectors (--sparse-mode bm25)
      pymilvus_bm25  – pre-computed PyMilvus BM25EmbeddingFunction vectors (--sparse-mode pymilvus_bm25)
      builtin_bm25   – Milvus built-in server-side BM25; raw text stored + queried
                       (pair with --sparse-mode bm25 for framework compat; text via --hf-dataset-id)
      splade         – pre-computed prithivida/Splade_PP_en_v1 vectors (--sparse-mode splade)
      milvus_splade  – pre-computed pymilvus SpladeEmbeddingFunction vectors (--sparse-mode milvus_splade)
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        milvus_sparse_mode: str = "splade",
        query_mode: str = "hybrid",
        ranker: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        drop_ratio_build: float = 0.2,
        drop_ratio_search: float = 0.0,
        inverted_index_algo: str = "DAAT_MAXSCORE",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        nlist: int = 128,
        nprobe: int = 10,
        ef_search: int = 128,
        ef_construction: int = 128,
        index_type: str = "HNSW",
    ):
        connect_kwargs = {"uri": uri}
        if token:
            connect_kwargs["token"] = token
        self.client = MilvusClient(**connect_kwargs)

        self.milvus_sparse_mode = milvus_sparse_mode
        self.query_mode         = query_mode
        self.ranker_type        = ranker
        self.rrf_k              = rrf_k
        self.dense_weight       = dense_weight
        self.sparse_weight      = sparse_weight
        self.drop_ratio_build    = drop_ratio_build
        self.drop_ratio_search   = drop_ratio_search
        self.inverted_index_algo = inverted_index_algo
        self.bm25_k1             = bm25_k1
        self.bm25_b              = bm25_b
        self.nlist               = nlist
        self.nprobe             = nprobe
        self.ef_search          = ef_search
        self.ef_construction    = ef_construction
        self.index_type         = index_type
        self.collection         = None
        self.metric_type        = "COSINE"

        logger.info("MilvusDB connected to %s (sparse_mode=%s)", uri, milvus_sparse_mode)

    # ── init ──────────────────────────────────────────────────────────────────

    def init(
        self,
        index_name: str,
        dimension: int = 384,
        space_type: str = "cosine",
        create: bool = True,
        **kwargs,
    ) -> None:
        self.collection  = index_name
        self.metric_type = _METRIC_MAP.get(space_type.lower(), "COSINE")

        if not create:
            logger.info("Connecting to existing collection '%s'", index_name)
            return

        existing = self.client.list_collections()
        if index_name in existing:
            logger.info("Collection '%s' already exists — skipping creation", index_name)
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",           DataType.VARCHAR,        max_length=256, is_primary=True)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR,   dim=dimension)

        if self.milvus_sparse_mode == "builtin_bm25":
            schema.add_field(
                "text",
                DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            )
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(Function(
                name="bm25_fn",
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            ))
        else:
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()

        # Dense index
        if self.index_type == "HNSW":
            dense_build_params = {"M": 16, "efConstruction": self.ef_construction}
        elif self.index_type in ("IVF_FLAT", "IVF_SQ8"):
            dense_build_params = {"nlist": self.nlist}
        else:
            dense_build_params = {}

        index_params.add_index(
            field_name="dense_vector",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=dense_build_params,
        )

        # Sparse index — BM25 metric for builtin_bm25, IP for all pre-computed modes
        sparse_metric = "BM25" if self.milvus_sparse_mode == "builtin_bm25" else "IP"
        sparse_params: dict = {
            "inverted_index_algo": self.inverted_index_algo,
            "drop_ratio_build":    self.drop_ratio_build,
        }
        if self.milvus_sparse_mode == "builtin_bm25":
            sparse_params["bm25_k1"] = self.bm25_k1
            sparse_params["bm25_b"]  = self.bm25_b
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type=sparse_metric,
            params=sparse_params,
        )

        self.client.create_collection(
            collection_name=index_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            "Created collection '%s' (dim=%d, metric=%s, sparse_mode=%s, index=%s)",
            index_name, dimension, self.metric_type, self.milvus_sparse_mode, self.index_type,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_sparse_dict(indices: List[int], values: List[float]) -> dict:
        return {int(i): float(v) for i, v in zip(indices, values)}

    def _dense_params(self) -> dict:
        return _dense_search_params(self.index_type, self.nprobe, self.ef_search)

    def _make_ranker(self):
        if self.ranker_type == "rrf":
            return RRFRanker(k=self.rrf_k)
        return WeightedRanker(self.dense_weight, self.sparse_weight)

    # ── index_batch ───────────────────────────────────────────────────────────

    def index_batch(self, points: List[Dict]) -> None:
        rows = []
        for p in points:
            if self.milvus_sparse_mode == "builtin_bm25":
                text = p.get("text", "")
                if not text:
                    continue
                rows.append({
                    "id":           str(p["id"]),
                    "dense_vector": [float(x) for x in p["vector"]],
                    "text":         text,
                })
            else:
                if not p["sparse_indices"]:
                    continue
                rows.append({
                    "id":            str(p["id"]),
                    "dense_vector":  [float(x) for x in p["vector"]],
                    "sparse_vector": self._to_sparse_dict(
                        p["sparse_indices"], p["sparse_values"]
                    ),
                })

        if not rows:
            return

        for attempt in range(MAX_RETRIES):
            try:
                self.client.insert(collection_name=self.collection, data=rows)
                return
            except Exception as e:
                logger.warning(
                    "index_batch attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.5)
                else:
                    raise

    # ── search ────────────────────────────────────────────────────────────────

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
                return self._hybrid_search(
                    dense_vector, sparse_indices, sparse_values, top_k, text
                )
            if self.query_mode == "sparse":
                return self._sparse_search(sparse_indices, sparse_values, top_k, text)
            return self._dense_search(dense_vector, top_k)
        except Exception as e:
            logger.error("search failed: %s", e)
            raise

    def _hybrid_search(
        self,
        dense_vector,
        sparse_indices,
        sparse_values,
        top_k,
        text,
    ) -> List[Dict]:
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": self.metric_type, "params": self._dense_params()},
            limit=top_k,
        )

        if self.milvus_sparse_mode == "builtin_bm25":
            sparse_req = AnnSearchRequest(
                data=[text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25", "params": {}},
                limit=top_k,
            )
        else:
            sparse_req = AnnSearchRequest(
                data=[self._to_sparse_dict(sparse_indices, sparse_values)],
                anns_field="sparse_vector",
                param={
                    "metric_type": "IP",
                    "params": {"drop_ratio_search": self.drop_ratio_search},
                },
                limit=top_k,
            )

        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=self._make_ranker(),
            limit=top_k,
        )
        return [{"id": str(hit["id"]), "score": hit["distance"]} for hit in results[0]]

    def _sparse_search(self, sparse_indices, sparse_values, top_k, text) -> List[Dict]:
        if self.milvus_sparse_mode == "builtin_bm25":
            data   = [text]
            metric = "BM25"
            params = {}
        else:
            data   = [self._to_sparse_dict(sparse_indices, sparse_values)]
            metric = "IP"
            params = {"drop_ratio_search": self.drop_ratio_search}

        results = self.client.search(
            collection_name=self.collection,
            data=data,
            anns_field="sparse_vector",
            search_params={"metric_type": metric, "params": params},
            limit=top_k,
        )
        return [{"id": str(hit["id"]), "score": hit["distance"]} for hit in results[0]]

    def _dense_search(self, dense_vector, top_k) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection,
            data=[dense_vector],
            anns_field="dense_vector",
            search_params={"metric_type": self.metric_type, "params": self._dense_params()},
            limit=top_k,
        )
        return [{"id": str(hit["id"]), "score": hit["distance"]} for hit in results[0]]

    # ── list_indices ──────────────────────────────────────────────────────────

    def list_indices(self) -> List[str]:
        return self.client.list_collections()

    # ── CLI plumbing ──────────────────────────────────────────────────────────

    @staticmethod
    def add_args(parser) -> None:
        g = parser.add_argument_group("Milvus options")
        g.add_argument(
            "--milvus-uri", default="http://localhost:19530",
            help="[Milvus] Server URI (default: http://localhost:19530)",
        )
        g.add_argument(
            "--milvus-token", default="",
            help="[Milvus] API token for Zilliz Cloud (default: '')",
        )
        g.add_argument(
            "--milvus-sparse-mode", default="splade",
            choices=["bm25", "pymilvus_bm25", "builtin_bm25", "splade", "milvus_splade"],
            help=(
                "[Milvus] Sparse mode:\n"
                "  bm25           – pre-computed rank_bm25 (--sparse-mode bm25)\n"
                "  pymilvus_bm25  – pre-computed PyMilvus BM25EmbeddingFunction (--sparse-mode pymilvus_bm25)\n"
                "  builtin_bm25   – Milvus server-side BM25 from raw text (--sparse-mode bm25 + --hf-dataset-id)\n"
                "  splade         – pre-computed prithivida/Splade_PP_en_v1 (--sparse-mode splade)\n"
                "  milvus_splade  – pre-computed pymilvus SpladeEmbeddingFunction (--sparse-mode milvus_splade)\n"
                "(default: splade)"
            ),
        )
        g.add_argument(
            "--milvus-query-mode", default="hybrid",
            choices=["hybrid", "sparse", "dense"],
            help="[Milvus] Query mode (default: hybrid)",
        )
        g.add_argument(
            "--milvus-ranker", default="rrf",
            choices=["rrf", "weighted"],
            help="[Milvus] Fusion ranker for hybrid search (default: rrf)",
        )
        g.add_argument(
            "--milvus-rrf-k", type=int, default=60,
            help="[Milvus] RRF k parameter (default: 60)",
        )
        g.add_argument(
            "--milvus-dense-weight", type=float, default=0.5,
            help="[Milvus] Dense weight for WeightedRanker (default: 0.5)",
        )
        g.add_argument(
            "--milvus-sparse-weight", type=float, default=0.5,
            help="[Milvus] Sparse weight for WeightedRanker (default: 0.5)",
        )
        g.add_argument(
            "--milvus-drop-ratio-build", type=float, default=0.2,
            help="[Milvus] Sparse SPARSE_INVERTED_INDEX drop_ratio_build (default: 0.2)",
        )
        g.add_argument(
            "--milvus-drop-ratio-search", type=float, default=0.0,
            help="[Milvus] Sparse search drop_ratio_search (default: 0.0)",
        )
        g.add_argument(
            "--milvus-inverted-index-algo", default="DAAT_MAXSCORE",
            choices=["DAAT_MAXSCORE", "DAAT_WAND", "TAAT_NAIVE"],
            help="[Milvus] Sparse inverted index algorithm (default: DAAT_MAXSCORE)",
        )
        g.add_argument(
            "--milvus-bm25-k1", type=float, default=1.2,
            help="[Milvus] BM25 k1 term-frequency saturation (builtin_bm25 only, default: 1.2)",
        )
        g.add_argument(
            "--milvus-bm25-b", type=float, default=0.75,
            help="[Milvus] BM25 b document-length normalization (builtin_bm25 only, default: 0.75)",
        )
        g.add_argument(
            "--milvus-nlist", type=int, default=128,
            help="[Milvus] IVF nlist (default: 128)",
        )
        g.add_argument(
            "--milvus-nprobe", type=int, default=10,
            help="[Milvus] IVF nprobe at search time (default: 10)",
        )
        g.add_argument(
            "--milvus-ef-search", type=int, default=128,
            help="[Milvus] HNSW ef at search time (default: 128)",
        )
        g.add_argument(
            "--milvus-ef-construction", type=int, default=128,
            help="[Milvus] HNSW efConstruction at build time (default: 128)",
        )
        g.add_argument(
            "--milvus-index-type", default="HNSW",
            choices=["FLAT", "IVF_FLAT", "IVF_SQ8", "HNSW"],
            help="[Milvus] Dense vector index type (default: IVF_FLAT)",
        )

    @staticmethod
    def build_config(args) -> dict:
        return {
            "uri":                args.milvus_uri,
            "token":              args.milvus_token,
            "milvus_sparse_mode": args.milvus_sparse_mode,
            "query_mode":         args.milvus_query_mode,
            "ranker":             args.milvus_ranker,
            "rrf_k":              args.milvus_rrf_k,
            "dense_weight":       args.milvus_dense_weight,
            "sparse_weight":      args.milvus_sparse_weight,
            "drop_ratio_build":    args.milvus_drop_ratio_build,
            "drop_ratio_search":   args.milvus_drop_ratio_search,
            "inverted_index_algo": args.milvus_inverted_index_algo,
            "bm25_k1":             args.milvus_bm25_k1,
            "bm25_b":              args.milvus_bm25_b,
            "nlist":              args.milvus_nlist,
            "nprobe":             args.milvus_nprobe,
            "ef_search":          args.milvus_ef_search,
            "ef_construction":    args.milvus_ef_construction,
            "index_type":         args.milvus_index_type,
        }
