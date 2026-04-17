# import datetime
# import logging
# import time
# from typing import Dict, List, Optional

# import requests
# from vespa.application import Vespa
# from vespa.package import (
#     HNSW,
#     ApplicationPackage,
#     Document,
#     Field,
#     Function,
#     GlobalPhaseRanking,
#     QueryField,
#     QueryProfile,
#     RankProfile,
#     Schema,
#     Validation,
#     ValidationID,
# )

# from src.interface import HybridDB

# logger = logging.getLogger(__name__)
# MAX_RETRIES = 10

# _DISTANCE_MAP = {
#     "cosine": "angular",
#     "dot":    "innerproduct",
#     "l2":     "euclidean",
# }

# _PRECISION_MAP = {
#     "float32":  "float",
#     "bfloat16": "bfloat16",
#     "int8":     "int8",
# }


# class VespaDB(HybridDB):
#     """
#     Vespa implementation of HybridDB.

#     Query modes
#     -----------
#     rrf (default)
#         Two queries (dense ANN + sparse full-scan), fused in Python with
#         Reciprocal Rank Fusion: score = 1/(k+dense_rank) + 1/(k+sparse_rank).
#         Comparable to Qdrant's FusionQuery(RRF). Use rrf_k=60 (standard).
#         NOTE: the sparse query is a full-scan, so this is best suited for
#         corpora up to ~50k docs. For larger corpora consider 'hybrid' mode.

#     native_rrf
#         Single dense-ANN query. Vespa computes RRF server-side in the global
#         phase using reciprocal_rank_fusion(closeness, sparse_score).
#         Faster than rrf (one round-trip). Candidates are ANN-only, so
#         sparse-only relevant docs may be missed. Use rerank_count >= 1000.

#     hybrid
#         Single dense-ANN query. Both dense (closeness) and sparse (dot-product)
#         scores are combined linearly in the rank profile:
#             alpha * closeness + (1 - alpha) * sparse_dot_product
#         Fast; single round-trip latency. Use alpha=0.5 for 50-50 weighting.

#     dense
#         Pure dense ANN retrieval (nearestNeighbor only).

#     sparse
#         Pure sparse full-scan scored by dot-product.

#     Apple-to-apple comparison with Endee
#     -------------------------------------
#     Run both with the SAME --sparse-mode (e.g. endee_bm25 or splade).
#     The framework pre-computes sparse vectors and saves them to .npy files;
#     both DBs receive identical dense + sparse input vectors.

#     Example (rrf mode, endee_bm25 sparse vectors):
#         python -m src.main --db vespa --vespa-query-mode rrf --vespa-rrf-k 60 \\
#             --sparse-mode endee_bm25 --index-name quora_bench ...

#     ef_search fix (vs db.py)
#     ------------------------
#     exploreAdditionalHits = max(0, ef_search - top_k)  [was: ef_search - candidate_k]

#     ef_search now controls HNSW exploration depth independently of the multiplier,
#     matching the vectorDBBench approach. targetHits=candidate_k is unchanged —
#     the reranking candidate pool size is still driven by the multiplier.
#     """

#     def __init__(
#         self,
#         url: str = "http://localhost",
#         port: int = 8080,
#         config_url: str = "",
#         config_port: int = 19071,
#         query_mode: str = "rrf",
#         target_hits_multiplier: int = 10,
#         rrf_k: int = 60,
#         alpha: float = 0.5,
#         ef_construction: int = 128,
#         ef_search: int = 128,
#         precision: str = "float32",
#         rerank_count: int = 1000,
#     ):
#         self.url = url
#         self.port = port
#         self.config_url = config_url if config_url else url
#         self.config_port = config_port
#         self.query_mode = query_mode
#         self.target_hits_multiplier = target_hits_multiplier
#         self.rrf_k = rrf_k
#         self.alpha = alpha
#         self.ef_construction = ef_construction
#         self.ef_search = ef_search
#         self.precision = precision
#         self.rerank_count = rerank_count
#         self.client: Optional[Vespa] = None
#         self.schema_name: Optional[str] = None
#         self.dimension: Optional[int] = None
#         self.distance_metric: str = "angular"

#     # ------------------------------------------------------------------
#     # HybridDB interface
#     # ------------------------------------------------------------------

#     def init(
#         self,
#         index_name: str,
#         dimension: int = 384,
#         space_type: str = "cosine",
#         create: bool = True,
#         **kwargs,
#     ) -> None:
#         self.schema_name = index_name
#         self.dimension = dimension
#         self.distance_metric = _DISTANCE_MAP.get(space_type.lower(), "angular")

#         if create:
#             app_package = self._build_application_package()
#             self._deploy(app_package)

#         self.client = Vespa(self.url, port=self.port)
#         self.client.wait_for_application_up(max_wait=300)
#         logger.info(
#             "VespaDB ready at %s:%d (schema=%s, mode=%s)",
#             self.url, self.port, self.schema_name, self.query_mode,
#         )

#     def index_batch(self, points: List[Dict]) -> None:
#         docs = []
#         for p in points:
#             if not p["sparse_indices"]:
#                 continue
#             fields = {
#                 "doc_id": str(p["id"]),
#                 "embedding": _scale_vector(p["vector"], self.precision),
#                 "sparse_embedding": _sparse_cells(
#                     p["sparse_indices"], p["sparse_values"]
#                 ),
#             }
#             if p.get("text"):
#                 fields["text"] = p["text"]
#             docs.append({"id": str(p["id"]), "fields": fields})

#         if not docs:
#             return

#         for attempt in range(MAX_RETRIES):
#             try:
#                 failures: List[str] = []

#                 def _cb(response, doc_id):
#                     if not response.is_successful():
#                         failures.append(doc_id)
#                         logger.warning(
#                             "Feed failed for doc %s: %s", doc_id, response.json
#                         )

#                 self.client.feed_iterable(
#                     iter(docs),
#                     schema=self.schema_name,
#                     namespace="hybrid",
#                     callback=_cb,
#                 )
#                 if failures:
#                     raise RuntimeError(
#                         f"Feed failed for {len(failures)} docs: {failures[:3]}"
#                     )
#                 return
#             except Exception as e:
#                 logger.warning(
#                     "index_batch attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e
#                 )
#                 if attempt < MAX_RETRIES - 1:
#                     time.sleep(1.5)
#                 else:
#                     logger.error(
#                         "index_batch failed after %d retries (%d points): %s",
#                         MAX_RETRIES, len(points), e,
#                     )
#                     raise

#     def search(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#         text: str = "",
#     ) -> List[Dict]:
#         dense_vector = _scale_vector(dense_vector, self.precision)
#         try:
#             if self.query_mode == "rrf":
#                 return self._search_rrf(
#                     dense_vector, sparse_indices, sparse_values, top_k
#                 )
#             if self.query_mode == "native_rrf":
#                 return self._search_native_rrf(
#                     dense_vector, sparse_indices, sparse_values, top_k
#                 )
#             if self.query_mode in ("bm25_text", "rrf_bm25_text", "hybrid_bm25_text"):
#                 return self._search_text(
#                     dense_vector, top_k, text
#                 )
#             body = self._build_single_query(
#                 dense_vector, sparse_indices, sparse_values, top_k
#             )
#             result = self.client.query(body)
#             children = result.get_json().get("root", {}).get("children", [])
#             return [
#                 {"id": child["fields"]["doc_id"], "score": child.get("relevance", 0.0)}
#                 for child in children
#             ]
#         except Exception as e:
#             logger.error("search failed: %s", e)
#             raise

#     def list_indices(self) -> List[str]:
#         if self.schema_name:
#             return [self.schema_name]
#         return []

#     # ------------------------------------------------------------------
#     # RRF search  (two queries, Python-level fusion)
#     # ------------------------------------------------------------------

#     def _search_rrf(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#     ) -> List[Dict]:
#         """
#         True Reciprocal Rank Fusion matching Qdrant's FusionQuery(RRF).

#         Formula (per document):
#             score = 1 / (rrf_k + dense_rank) + 1 / (rrf_k + sparse_rank)

#         Dense rank  : position in nearestNeighbor results (0-indexed)
#         Sparse rank : position in sparse dot-product results (0-indexed)
#         Unranked    : treated as rank = candidate_pool_size (no contribution)
#         """
#         candidate_k = top_k * self.target_hits_multiplier

#         # --- dense query ---
#         # FIX: exploreAdditionalHits based on top_k (not candidate_k) so
#         # ef_search controls HNSW exploration depth independently of the multiplier.
#         # Total explored = candidate_k + max(0, ef_search - top_k)
#         explore_additional = max(0, self.ef_search - top_k)
#         dense_body = {
#             "yql": (
#                 f"select doc_id from {self.schema_name} where "
#                 f"{{targetHits: {candidate_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
#                 f"nearestNeighbor(embedding, q_embedding)"
#             ),
#             "hits": candidate_k,
#             "input.query(q_embedding)": dense_vector,
#             "ranking": "dense",
#         }
#         dense_hits = self._run_query(dense_body)

#         # --- sparse query (full-scan, doc-set scored by dot product) ---
#         sparse_body = {
#             "yql": f"select doc_id from {self.schema_name} where true",
#             "hits": candidate_k,
#             "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#             "ranking": "sparse",
#         }
#         sparse_hits = self._run_query(sparse_body)

#         # --- RRF fusion ---
#         return _rrf_fuse(dense_hits, sparse_hits, k=self.rrf_k, top_k=top_k)

#     # ------------------------------------------------------------------
#     # Native RRF  (single query, Vespa global-phase reciprocal_rank_fusion)
#     # ------------------------------------------------------------------

#     def _search_native_rrf(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#     ) -> List[Dict]:
#         """
#         Vespa-native RRF via GlobalPhaseRanking with reciprocal_rank_fusion().

#         Single ANN query retrieves candidate_k docs; Vespa re-ranks them
#         server-side using RRF of (closeness, sparse_score). Faster than the
#         two-query Python RRF but only considers ANN-retrieved candidates.
#         """
#         candidate_k = min(top_k * self.target_hits_multiplier, self.rerank_count)
#         # FIX: same as rrf — ef_search relative to top_k, not candidate_k
#         explore_additional = max(0, self.ef_search - top_k)
#         body = {
#             "yql": (
#                 f"select doc_id from {self.schema_name} where "
#                 f"{{targetHits: {candidate_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
#                 f"nearestNeighbor(embedding, q_embedding)"
#             ),
#             "hits": top_k,
#             "input.query(q_embedding)": dense_vector,
#             "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#             "ranking": "native_rrf",
#         }
#         return self._run_query(body)

#     # ------------------------------------------------------------------
#     # Native BM25 text modes
#     # ------------------------------------------------------------------

#     def _search_text(
#         self,
#         dense_vector: List[float],
#         top_k: int,
#         text: str,
#     ) -> List[Dict]:
#         """
#         Three sub-modes driven by self.query_mode:

#         bm25_text
#             Pure Vespa BM25 full-text scan. No dense vector used.
#             YQL: userQuery() over the indexed 'text' field.

#         rrf_bm25_text
#             Single query: ANN OR text matches fetched together.
#             Global phase applies reciprocal_rank_fusion(closeness, bm25(text)).
#             True hybrid — both dense and text candidates included.

#         hybrid_bm25_text
#             Single dense-ANN query. Linear combination in rank profile:
#                 alpha * closeness + (1-alpha) * bm25(text)
#         """
#         candidate_k = top_k * self.target_hits_multiplier
#         # FIX: ef_search now applied to rrf_bm25_text and hybrid_bm25_text ANN legs
#         explore_additional = max(0, self.ef_search - top_k)

#         if self.query_mode == "bm25_text":
#             body = {
#                 "yql": f"select doc_id from {self.schema_name} where userQuery()",
#                 "query": text,
#                 "hits": top_k,
#                 "ranking": "bm25_text",
#             }
#         elif self.query_mode == "rrf_bm25_text":
#             body = {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {candidate_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
#                     f"nearestNeighbor(embedding, q_embedding)"
#                     f" OR userQuery()"
#                 ),
#                 "query": text,
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "ranking": "rrf_bm25_text",
#             }
#         else:  # hybrid_bm25_text
#             body = {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {candidate_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
#                     f"nearestNeighbor(embedding, q_embedding)"
#                     f" OR userQuery()"
#                 ),
#                 "query": text,
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "input.query(alpha)": self.alpha,
#                 "ranking": "hybrid_bm25_text",
#             }
#         return self._run_query(body)

#     def _run_query(self, body: dict) -> List[Dict]:
#         result = self.client.query(body)
#         children = result.get_json().get("root", {}).get("children", [])
#         return [
#             {"id": child["fields"]["doc_id"], "score": child.get("relevance", 0.0)}
#             for child in children
#         ]

#     # ------------------------------------------------------------------
#     # Single-query modes  (hybrid / dense / sparse)
#     # ------------------------------------------------------------------

#     def _build_single_query(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#     ) -> dict:
#         candidate_hits = top_k * self.target_hits_multiplier

#         if self.query_mode == "hybrid":
#             return {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {candidate_hits}}}"
#                     f"nearestNeighbor(embedding, q_embedding)"
#                 ),
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#                 "input.query(alpha)": self.alpha,
#                 "ranking": "hybrid",
#             }

#         if self.query_mode == "dense":
#             return {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {top_k}}}"
#                     f"nearestNeighbor(embedding, q_embedding)"
#                 ),
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "ranking": "dense",
#             }

#         # sparse — full-scan scored by sparse dot product
#         return {
#             "yql": f"select doc_id from {self.schema_name} where true",
#             "hits": top_k,
#             "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#             "ranking": "sparse",
#         }

#     # ------------------------------------------------------------------
#     # Application package
#     # ------------------------------------------------------------------

#     def _build_application_package(self) -> ApplicationPackage:
#         fields = [
#             Field(
#                 name="doc_id",
#                 type="string",
#                 indexing=["summary", "attribute"],
#             ),
#             Field(
#                 name="embedding",
#                 type=f"tensor<float>(x[{self.dimension}])",
#                 indexing=["summary", "attribute", "index"],
#                 ann=HNSW(
#                     distance_metric=self.distance_metric,
#                     max_links_per_node=16,
#                     neighbors_to_explore_at_insert=self.ef_construction,
#                 ),
#             ),
#             Field(
#                 name="sparse_embedding",
#                 type="tensor<float>(x{})",
#                 indexing=["summary", "attribute"],
#             ),
#             Field(
#                 name="text",
#                 type="string",
#                 indexing=["index", "summary"],
#                 index="enable-bm25",
#             ),
#         ]

#         rank_profiles = [
#             # Used by rrf mode (dense leg) and hybrid mode
#             RankProfile(
#                 name="dense",
#                 first_phase="closeness(field, embedding)",
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                 ],
#             ),
#             # Used by rrf mode (sparse leg) and standalone sparse mode
#             RankProfile(
#                 name="sparse",
#                 first_phase="sum(query(q_sparse) * attribute(sparse_embedding))",
#                 inputs=[
#                     ("query(q_sparse)", "tensor<float>(x{})"),
#                 ],
#             ),
#             # Single-query linear combination (alpha * dense + (1-alpha) * sparse)
#             RankProfile(
#                 name="hybrid",
#                 first_phase=(
#                     "query(alpha) * closeness(field, embedding) + "
#                     "(1 - query(alpha)) * sum(query(q_sparse) * attribute(sparse_embedding))"
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                     ("query(q_sparse)", "tensor<float>(x{})"),
#                     ("query(alpha)", "double"),
#                 ],
#             ),
#             # Pure native BM25 text search (no dense)
#             RankProfile(
#                 name="bm25_text",
#                 first_phase="bm25(text)",
#             ),
#             # Native RRF: ANN OR text candidates, global-phase RRF(closeness, bm25)
#             RankProfile(
#                 name="rrf_bm25_text",
#                 first_phase="closeness(field, embedding) + bm25(text)",
#                 global_phase=GlobalPhaseRanking(
#                     expression="reciprocal_rank_fusion(closeness(field, embedding), bm25(text))",
#                     rerank_count=self.rerank_count,
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                 ],
#             ),
#             # Linear hybrid: alpha*closeness + (1-alpha)*bm25(text)
#             RankProfile(
#                 name="hybrid_bm25_text",
#                 first_phase=(
#                     "query(alpha) * closeness(field, embedding) + "
#                     "(1 - query(alpha)) * bm25(text)"
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                     ("query(alpha)", "double"),
#                 ],
#             ),
#             # Vespa-native RRF: single ANN query, server-side global-phase fusion
#             RankProfile(
#                 name="native_rrf",
#                 first_phase="closeness(field, embedding)",
#                 functions=[
#                     Function(
#                         name="sparse_score",
#                         expression="sum(query(q_sparse) * attribute(sparse_embedding))",
#                     ),
#                 ],
#                 global_phase=GlobalPhaseRanking(
#                     expression="reciprocal_rank_fusion(closeness(field, embedding), sparse_score)",
#                     rerank_count=self.rerank_count,
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                     ("query(q_sparse)", "tensor<float>(x{})"),
#                 ],
#             ),
#         ]

#         tomorrow = datetime.date.today() + datetime.timedelta(days=1)

#         return ApplicationPackage(
#             name="hybridbench",
#             schema=[
#                 Schema(
#                     name=self.schema_name,
#                     document=Document(fields=fields),
#                     rank_profiles=rank_profiles,
#                 )
#             ],
#             query_profile=QueryProfile(
#                 fields=[QueryField(name="maxHits", value=10000)]
#             ),
#             validations=[
#                 Validation(ValidationID.tensorTypeChange, until=tomorrow),
#                 Validation(ValidationID.fieldTypeChange, until=tomorrow),
#                 Validation(ValidationID.contentClusterRemoval, until=tomorrow),
#                 Validation(ValidationID.contentTypeRemoval, until=tomorrow),
#             ],
#         )

#     def _deploy(self, app_package: ApplicationPackage) -> None:
#         deploy_url = (
#             f"{self.config_url}:{self.config_port}"
#             "/application/v2/tenant/default/prepareandactivate"
#         )
#         package_data = app_package.to_zip()
#         response = requests.post(
#             url=deploy_url,
#             data=package_data,
#             headers={"Content-Type": "application/zip"},
#             timeout=120,
#         )
#         if not response.ok:
#             logger.error(
#                 "Deploy failed %d: %s", response.status_code, response.text
#             )
#         response.raise_for_status()
#         logger.info(
#             "Deployed Vespa application 'hybridbench' with schema '%s'",
#             self.schema_name,
#         )

#     # ------------------------------------------------------------------
#     # CLI integration
#     # ------------------------------------------------------------------

#     @staticmethod
#     def add_args(parser) -> None:
#         g = parser.add_argument_group("Vespa options")
#         g.add_argument(
#             "--vespa-url", default="http://localhost",
#             help="[Vespa] Base URL without port (default: http://localhost)",
#         )
#         g.add_argument(
#             "--vespa-port", type=int, default=8080,
#             help="[Vespa] Query/feed port (default: 8080)",
#         )
#         g.add_argument(
#             "--vespa-config-url", default="",
#             help="[Vespa] Config server URL (default: same as --vespa-url)",
#         )
#         g.add_argument(
#             "--vespa-config-port", type=int, default=19071,
#             help="[Vespa] Config server port (default: 19071)",
#         )
#         g.add_argument(
#             "--vespa-query-mode", default="rrf",
#             choices=["rrf", "native_rrf", "hybrid", "dense", "sparse",
#                      "bm25_text", "rrf_bm25_text", "hybrid_bm25_text"],
#             help=(
#                 "[Vespa] Query mode: "
#                 "rrf = two-query Python RRF (comparable to Qdrant RRF, best recall); "
#                 "native_rrf = Vespa server-side RRF with pre-computed sparse; "
#                 "hybrid = linear combination alpha*dense + (1-alpha)*sparse; "
#                 "dense = ANN only; sparse = full-scan dot-product; "
#                 "bm25_text = Vespa native BM25 on raw text (requires --hf-dataset-id); "
#                 "rrf_bm25_text = RRF(closeness, bm25) on ANN+text candidates; "
#                 "hybrid_bm25_text = alpha*closeness + (1-alpha)*bm25(text) "
#                 "(default: rrf)"
#             ),
#         )
#         g.add_argument(
#             "--vespa-rrf-k", type=int, default=60,
#             help="[Vespa] RRF constant k in 1/(k+rank). Only used in rrf mode (default: 60)",
#         )
#         g.add_argument(
#             "--vespa-target-hits-multiplier", type=int, default=10,
#             help="[Vespa] Candidate pool = top_k * multiplier for dense/rrf queries (default: 10)",
#         )
#         g.add_argument(
#             "--vespa-alpha", type=float, default=0.5,
#             help="[Vespa] Dense weight for hybrid mode: alpha*dense + (1-alpha)*sparse (default: 0.5)",
#         )
#         g.add_argument(
#             "--vespa-ef-construction", type=int, default=128,
#             help="[Vespa] HNSW ef_construction: neighbors explored at insert time (default: 128)",
#         )
#         g.add_argument(
#             "--vespa-ef-search", type=int, default=128,
#             help="[Vespa] HNSW ef_search: min candidates explored at query time (default: 128)",
#         )
#         g.add_argument(
#             "--vespa-precision", default="float32",
#             choices=["float32", "bfloat16", "int8"],
#             help="[Vespa] Vector precision: float32 (4B), bfloat16 (2B), int8 (1B) (default: float32)",
#         )
#         g.add_argument(
#             "--vespa-rerank-count", type=int, default=1000,
#             help="[Vespa] GlobalPhaseRanking rerank_count: candidates reranked server-side (default: 1000)",
#         )

#     @staticmethod
#     def build_config(args) -> dict:
#         return {
#             "url":                    args.vespa_url,
#             "port":                   args.vespa_port,
#             "config_url":             args.vespa_config_url,
#             "config_port":            args.vespa_config_port,
#             "query_mode":             args.vespa_query_mode,
#             "rrf_k":                  args.vespa_rrf_k,
#             "target_hits_multiplier": args.vespa_target_hits_multiplier,
#             "alpha":                  args.vespa_alpha,
#             "ef_construction":        args.vespa_ef_construction,
#             "ef_search":              args.vespa_ef_search,
#             "precision":              args.vespa_precision,
#             "rerank_count":           args.vespa_rerank_count,
#         }


# # ------------------------------------------------------------------
# # Module-level helpers
# # ------------------------------------------------------------------

# def _scale_vector(vector: List[float], precision: str) -> List[float]:
#     """
#     Scale a float32 vector for int8 storage/query.

#     int8 range is [-128, 127]. Standard embeddings are in [-1, 1].
#     Multiply by 127 to map [-1, 1] → [-127, 127].
#     For float32 and bfloat16, no scaling needed — returns vector unchanged.
#     """
#     if precision == "int8":
#         return [float(v * 127) for v in vector]
#     return vector


# def _sparse_cells(indices: List[int], values: List[float]) -> dict:
#     """
#     Format a SPLADE/BM25 sparse vector as Vespa's mapped-tensor JSON format.

#     Vespa tensor<float>(x{}) expects:
#         {"cells": {"dimension_index_as_string": float_value, ...}}
#     """
#     return {"cells": {str(i): float(v) for i, v in zip(indices, values)}}


# def _rrf_fuse(
#     dense_hits: List[Dict],
#     sparse_hits: List[Dict],
#     k: int,
#     top_k: int,
# ) -> List[Dict]:
#     """
#     Reciprocal Rank Fusion matching Qdrant's FusionQuery(RRF).

#         score(doc) = 1/(k + dense_rank) + 1/(k + sparse_rank)

#     Documents absent from a list get rank = len(list), contributing
#     minimally (same as Qdrant's unranked-document treatment).
#     """
#     pool_size = max(len(dense_hits), len(sparse_hits))

#     dense_rank = {h["id"]: i for i, h in enumerate(dense_hits)}
#     sparse_rank = {h["id"]: i for i, h in enumerate(sparse_hits)}

#     all_ids = set(dense_rank) | set(sparse_rank)
#     scores: Dict[str, float] = {}
#     for doc_id in all_ids:
#         dr = dense_rank.get(doc_id, pool_size)
#         sr = sparse_rank.get(doc_id, pool_size)
#         scores[doc_id] = 1.0 / (k + dr) + 1.0 / (k + sr)

#     ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     return [{"id": doc_id, "score": score} for doc_id, score in ranked[:top_k]]

#------------------------------------------------------------------------------------

import datetime
import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests
from vespa.application import Vespa
from vespa.package import (
    HNSW,
    ApplicationPackage,
    Document,
    Field,
    Function,
    GlobalPhaseRanking,
    QueryField,
    QueryProfile,
    RankProfile,
    Schema,
    Validation,
    ValidationID,
)

from src.interface import HybridDB

logger = logging.getLogger(__name__)
MAX_RETRIES = 10

_DISTANCE_MAP = {
    "cosine": "angular",
    "dot":    "innerproduct",
    "l2":     "euclidean",
}

_PRECISION_MAP = {
    "float32":  "float",
    "bfloat16": "bfloat16",
    "int8":     "int8",
}

_SCHEMA_REGISTRY_PATH = "vespa_schema_registry.json"


class VespaDB(HybridDB):
    """
    Vespa implementation of HybridDB.

    Query modes
    -----------
    rrf (default)
        Two queries (dense ANN + sparse full-scan), fused in Python with
        Reciprocal Rank Fusion: score = 1/(k+dense_rank) + 1/(k+sparse_rank).
        Comparable to Qdrant's FusionQuery(RRF). Use rrf_k=60 (standard).
        NOTE: the sparse query is a full-scan, so this is best suited for
        corpora up to ~50k docs. For larger corpora consider 'hybrid' mode.

    native_rrf
        Single dense-ANN query. Vespa computes RRF server-side in the global
        phase using reciprocal_rank_fusion(closeness, sparse_score).
        Faster than rrf (one round-trip). Candidates are ANN-only, so
        sparse-only relevant docs may be missed. Use rerank_count >= 1000.

    hybrid
        Single dense-ANN query. Both dense (closeness) and sparse (dot-product)
        scores are combined linearly in the rank profile:
            alpha * closeness + (1 - alpha) * sparse_dot_product
        Fast; single round-trip latency. Use alpha=0.5 for 50-50 weighting.

    dense
        Pure dense ANN retrieval (nearestNeighbor only).

    sparse
        Pure sparse full-scan scored by dot-product.

    Apple-to-apple comparison with Endee
    -------------------------------------
    Run both with the SAME --sparse-mode (e.g. endee_bm25 or splade).
    The framework pre-computes sparse vectors and saves them to .npy files;
    both DBs receive identical dense + sparse input vectors.

    Example (rrf mode, endee_bm25 sparse vectors):
        python -m src.main --db vespa --vespa-query-mode rrf --vespa-rrf-k 60 \\
            --sparse-mode endee_bm25 --index-name quora_bench ...
    """

    def __init__(
        self,
        url: str = "http://localhost",
        port: int = 8080,
        config_url: str = "",
        config_port: int = 19071,
        query_mode: str = "rrf",
        target_hits_multiplier: int = 10,
        rrf_k: int = 60,
        alpha: float = 0.5,
        ef_construction: int = 128,
        ef_search: int = 128,
        precision: str = "float32",
        rerank_count: int = 1000,
    ):
        self.url = url
        self.port = port
        self.config_url = config_url if config_url else url
        self.config_port = config_port
        self.query_mode = query_mode
        self.target_hits_multiplier = target_hits_multiplier
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.precision = precision
        self.rerank_count = rerank_count
        self.client: Optional[Vespa] = None
        self.schema_name: Optional[str] = None
        self.dimension: Optional[int] = None
        self.distance_metric: str = "angular"

    # ------------------------------------------------------------------
    # HybridDB interface
    # ------------------------------------------------------------------

    def init(
        self,
        index_name: str,
        dimension: int = 384,
        space_type: str = "cosine",
        create: bool = True,
        **kwargs,
    ) -> None:
        self.schema_name = index_name
        self.dimension = dimension
        self.distance_metric = _DISTANCE_MAP.get(space_type.lower(), "angular")

        if create:
            self._save_to_registry()
            app_package = self._build_application_package()
            self._deploy(app_package)

        self.client = Vespa(self.url, port=self.port)
        self.client.wait_for_application_up(max_wait=300)
        logger.info(
            "VespaDB ready at %s:%d (schema=%s, mode=%s)",
            self.url, self.port, self.schema_name, self.query_mode,
        )

    def index_batch(self, points: List[Dict]) -> None:
        docs = []
        for p in points:
            if not p["sparse_indices"]:
                continue
            fields = {
                "doc_id": str(p["id"]),
                "embedding": _scale_vector(p["vector"], self.precision),
                "sparse_embedding": _sparse_cells(
                    p["sparse_indices"], p["sparse_values"]
                ),
            }
            if p.get("text"):
                fields["text"] = p["text"]
            docs.append({"id": str(p["id"]), "fields": fields})

        if not docs:
            return

        for attempt in range(MAX_RETRIES):
            try:
                failures: List[str] = []

                def _cb(response, doc_id):
                    if not response.is_successful():
                        failures.append(doc_id)
                        logger.warning(
                            "Feed failed for doc %s: %s", doc_id, response.json
                        )

                self.client.feed_iterable(
                    iter(docs),
                    schema=self.schema_name,
                    namespace="hybrid",
                    callback=_cb,
                )
                if failures:
                    raise RuntimeError(
                        f"Feed failed for {len(failures)} docs: {failures[:3]}"
                    )
                return
            except Exception as e:
                logger.warning(
                    "index_batch attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.5)
                else:
                    logger.error(
                        "index_batch failed after %d retries (%d points): %s",
                        MAX_RETRIES, len(points), e,
                    )
                    raise

    def search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
        text: str = "",
    ) -> List[Dict]:
        dense_vector = _scale_vector(dense_vector, self.precision)
        try:
            if self.query_mode == "rrf":
                return self._search_rrf(
                    dense_vector, sparse_indices, sparse_values, top_k
                )
            if self.query_mode == "native_rrf":
                return self._search_native_rrf(
                    dense_vector, sparse_indices, sparse_values, top_k
                )
            if self.query_mode in ("bm25_text", "rrf_bm25_text", "hybrid_bm25_text"):
                return self._search_text(
                    dense_vector, top_k, text
                )
            body = self._build_single_query(
                dense_vector, sparse_indices, sparse_values, top_k
            )
            result = self.client.query(body)
            children = result.get_json().get("root", {}).get("children", [])
            return [
                {"id": child["fields"]["doc_id"], "score": child.get("relevance", 0.0)}
                for child in children
            ]
        except Exception as e:
            logger.error("search failed: %s", e)
            raise

    def list_indices(self) -> List[str]:
        if self.schema_name:
            return [self.schema_name]
        return []

    # ------------------------------------------------------------------
    # RRF search  (two queries, Python-level fusion)
    # ------------------------------------------------------------------

    def _search_rrf(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
    ) -> List[Dict]:
        """
        True Reciprocal Rank Fusion matching Qdrant's FusionQuery(RRF).

        Formula (per document):
            score = 1 / (rrf_k + dense_rank) + 1 / (rrf_k + sparse_rank)

        Dense rank  : position in nearestNeighbor results (0-indexed)
        Sparse rank : position in sparse dot-product results (0-indexed)
        Unranked    : treated as rank = candidate_pool_size (no contribution)
        """
        candidate_k = top_k * self.target_hits_multiplier

        # --- dense query ---
        explore_additional = max(0, self.ef_search - candidate_k)
        dense_body = {
            "yql": (
                f"select doc_id from {self.schema_name} where "
                f"{{targetHits: {candidate_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
                f"nearestNeighbor(embedding, q_embedding)"
            ),
            "hits": candidate_k,
            "input.query(q_embedding)": dense_vector,
            "ranking": "dense",
        }
        dense_hits = self._run_query(dense_body)

        # --- sparse query (full-scan, doc-set scored by dot product) ---
        sparse_body = {
            "yql": f"select doc_id from {self.schema_name} where true",
            "hits": candidate_k,
            "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
            "ranking": "sparse",
        }
        sparse_hits = self._run_query(sparse_body)

        # --- RRF fusion ---
        return _rrf_fuse(dense_hits, sparse_hits, k=self.rrf_k, top_k=top_k)

    # ------------------------------------------------------------------
    # Native RRF  (single query, Vespa global-phase reciprocal_rank_fusion)
    # ------------------------------------------------------------------

    def _search_native_rrf(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
    ) -> List[Dict]:
        """
        Vespa-native RRF via GlobalPhaseRanking with reciprocal_rank_fusion().

        Single ANN query retrieves candidate_k docs; Vespa re-ranks them
        server-side using RRF of (closeness, sparse_score). Faster than the
        two-query Python RRF but only considers ANN-retrieved candidates.
        """
        candidate_k = min(top_k * self.target_hits_multiplier, self.rerank_count)
        explore_additional = max(0, self.ef_search - candidate_k)
        body = {
            "yql": (
                f"select doc_id from {self.schema_name} where "
                f"{{targetHits: {candidate_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
                f"nearestNeighbor(embedding, q_embedding)"
            ),
            "hits": top_k,
            "input.query(q_embedding)": dense_vector,
            "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
            "ranking": "native_rrf",
        }
        return self._run_query(body)

    # ------------------------------------------------------------------
    # Native BM25 text modes
    # ------------------------------------------------------------------

    def _search_text(
        self,
        dense_vector: List[float],
        top_k: int,
        text: str,
    ) -> List[Dict]:
        """
        Three sub-modes driven by self.query_mode:

        bm25_text
            Pure Vespa BM25 full-text scan. No dense vector used.
            YQL: userQuery() over the indexed 'text' field.

        rrf_bm25_text
            Single query: ANN OR text matches fetched together.
            Global phase applies reciprocal_rank_fusion(closeness, bm25(text)).
            True hybrid — both dense and text candidates included.

        hybrid_bm25_text
            Single dense-ANN query. Linear combination in rank profile:
                alpha * closeness + (1-alpha) * bm25(text)
        """
        # VectorDB Bench-equivalent: targetHits=top_k, exploreAdditionalHits=max(0, ef_search-top_k)
        # Total HNSW exploration = top_k + max(0, ef_search - top_k) = max(ef_search, top_k)
        # ef_search is the single recall knob, comparable to other DBs.
        explore_additional = max(0, self.ef_search - top_k)

        if self.query_mode == "bm25_text":
            body = {
                "yql": f"select doc_id from {self.schema_name} where userQuery()",
                "query": text,
                "hits": top_k,
                "ranking": "bm25_text",
            }
        elif self.query_mode == "rrf_bm25_text":
            body = {
                "yql": (
                    f"select doc_id from {self.schema_name} where "
                    f"{{targetHits: {top_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
                    f"nearestNeighbor(embedding, q_embedding)"
                    f" OR userQuery()"
                ),
                "query": text,
                "hits": top_k,
                "input.query(q_embedding)": dense_vector,
                "ranking": "rrf_bm25_text",
            }
        else:  # hybrid_bm25_text
            body = {
                "yql": (
                    f"select doc_id from {self.schema_name} where "
                    f"{{targetHits: {top_k}, hnsw.exploreAdditionalHits: {explore_additional}}}"
                    f"nearestNeighbor(embedding, q_embedding)"
                    f" OR userQuery()"
                ),
                "query": text,
                "hits": top_k,
                "input.query(q_embedding)": dense_vector,
                "input.query(alpha)": self.alpha,
                "ranking": "hybrid_bm25_text",
            }
        return self._run_query(body)

    def _run_query(self, body: dict) -> List[Dict]:
        result = self.client.query(body)
        children = result.get_json().get("root", {}).get("children", [])
        return [
            {"id": child["fields"]["doc_id"], "score": child.get("relevance", 0.0)}
            for child in children
        ]

    # ------------------------------------------------------------------
    # Single-query modes  (hybrid / dense / sparse)
    # ------------------------------------------------------------------

    def _build_single_query(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
    ) -> dict:
        candidate_hits = top_k * self.target_hits_multiplier

        if self.query_mode == "hybrid":
            return {
                "yql": (
                    f"select doc_id from {self.schema_name} where "
                    f"{{targetHits: {candidate_hits}}}"
                    f"nearestNeighbor(embedding, q_embedding)"
                ),
                "hits": top_k,
                "input.query(q_embedding)": dense_vector,
                "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
                "input.query(alpha)": self.alpha,
                "ranking": "hybrid",
            }

        if self.query_mode == "dense":
            return {
                "yql": (
                    f"select doc_id from {self.schema_name} where "
                    f"{{targetHits: {top_k}}}"
                    f"nearestNeighbor(embedding, q_embedding)"
                ),
                "hits": top_k,
                "input.query(q_embedding)": dense_vector,
                "ranking": "dense",
            }

        # sparse — full-scan scored by sparse dot product
        return {
            "yql": f"select doc_id from {self.schema_name} where true",
            "hits": top_k,
            "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
            "ranking": "sparse",
        }

    # ------------------------------------------------------------------
    # Schema registry  (keeps all deployed schemas alive across deploys)
    # ------------------------------------------------------------------

    def _save_to_registry(self) -> None:
        registry = self._load_registry()
        registry[self.schema_name] = {
            "dimension":      self.dimension,
            "distance_metric": self.distance_metric,
            "ef_construction": self.ef_construction,
            "rerank_count":   self.rerank_count,
            "precision":      self.precision,
        }
        with open(_SCHEMA_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info("Registry updated: %s", list(registry.keys()))

    def _load_registry(self) -> dict:
        if not os.path.exists(_SCHEMA_REGISTRY_PATH):
            return {}
        with open(_SCHEMA_REGISTRY_PATH) as f:
            return json.load(f)

    def _build_schema(self, schema_name: str, cfg: dict) -> "Schema":
        dimension      = cfg["dimension"]
        distance_metric = cfg["distance_metric"]
        ef_construction = cfg["ef_construction"]
        rerank_count   = cfg["rerank_count"]
        precision      = cfg.get("precision", "float32")
        vespa_type     = _PRECISION_MAP.get(precision, "float")

        fields = [
            Field(name="doc_id", type="string", indexing=["summary", "attribute"]),
            Field(
                name="embedding",
                type=f"tensor<{vespa_type}>(x[{dimension}])",
                indexing=["summary", "attribute", "index"],
                ann=HNSW(
                    distance_metric=distance_metric,
                    max_links_per_node=16,
                    neighbors_to_explore_at_insert=ef_construction,
                ),
            ),
            Field(
                name="sparse_embedding",
                type="tensor<float>(x{})",
                indexing=["summary", "attribute"],
            ),
            Field(
                name="text",
                type="string",
                indexing=["index", "summary"],
                index="enable-bm25",
            ),
        ]

        rank_profiles = [
            RankProfile(
                name="dense",
                first_phase="closeness(field, embedding)",
                inputs=[("query(q_embedding)", f"tensor<float>(x[{dimension}])")],
            ),
            RankProfile(
                name="sparse",
                first_phase="sum(query(q_sparse) * attribute(sparse_embedding))",
                inputs=[("query(q_sparse)", "tensor<float>(x{})")],
            ),
            RankProfile(
                name="hybrid",
                first_phase=(
                    "query(alpha) * closeness(field, embedding) + "
                    "(1 - query(alpha)) * sum(query(q_sparse) * attribute(sparse_embedding))"
                ),
                inputs=[
                    ("query(q_embedding)", f"tensor<float>(x[{dimension}])"),
                    ("query(q_sparse)", "tensor<float>(x{})"),
                    ("query(alpha)", "double"),
                ],
            ),
            RankProfile(name="bm25_text", first_phase="bm25(text)"),
            RankProfile(
                name="rrf_bm25_text",
                first_phase="closeness(field, embedding) + bm25(text)",
                global_phase=GlobalPhaseRanking(
                    expression="reciprocal_rank_fusion(closeness(field, embedding), bm25(text))",
                    rerank_count=rerank_count,
                ),
                inputs=[("query(q_embedding)", f"tensor<float>(x[{dimension}])")],
            ),
            RankProfile(
                name="hybrid_bm25_text",
                first_phase=(
                    "query(alpha) * closeness(field, embedding) + "
                    "(1 - query(alpha)) * bm25(text)"
                ),
                inputs=[
                    ("query(q_embedding)", f"tensor<float>(x[{dimension}])"),
                    ("query(alpha)", "double"),
                ],
            ),
            RankProfile(
                name="native_rrf",
                first_phase="closeness(field, embedding)",
                functions=[
                    Function(
                        name="sparse_score",
                        expression="sum(query(q_sparse) * attribute(sparse_embedding))",
                    ),
                ],
                global_phase=GlobalPhaseRanking(
                    expression="reciprocal_rank_fusion(closeness(field, embedding), sparse_score)",
                    rerank_count=rerank_count,
                ),
                inputs=[
                    ("query(q_embedding)", f"tensor<float>(x[{dimension}])"),
                    ("query(q_sparse)", "tensor<float>(x{})"),
                ],
            ),
        ]

        return Schema(
            name=schema_name,
            document=Document(fields=fields),
            rank_profiles=rank_profiles,
        )

    # ------------------------------------------------------------------
    # Application package
    # ------------------------------------------------------------------

    def _build_application_package(self) -> ApplicationPackage:
        registry = self._load_registry()
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)

        schemas = [
            self._build_schema(name, cfg)
            for name, cfg in registry.items()
        ]

        return ApplicationPackage(
            name="hybridbench",
            schema=schemas,
            query_profile=QueryProfile(
                fields=[QueryField(name="maxHits", value=10000)]
            ),
            validations=[
                Validation(ValidationID.tensorTypeChange, until=tomorrow),
                Validation(ValidationID.fieldTypeChange, until=tomorrow),
                Validation(ValidationID.contentClusterRemoval, until=tomorrow),
                Validation(ValidationID.contentTypeRemoval, until=tomorrow),
            ],
        )

    def _deploy(self, app_package: ApplicationPackage) -> None:
        deploy_url = (
            f"{self.config_url}:{self.config_port}"
            "/application/v2/tenant/default/prepareandactivate"
        )
        package_data = app_package.to_zip()
        response = requests.post(
            url=deploy_url,
            data=package_data,
            headers={"Content-Type": "application/zip"},
            timeout=120,
        )
        if not response.ok:
            logger.error(
                "Deploy failed %d: %s", response.status_code, response.text
            )
        response.raise_for_status()
        logger.info(
            "Deployed Vespa application 'hybridbench' with schema '%s'",
            self.schema_name,
        )

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    @staticmethod
    def add_args(parser) -> None:
        g = parser.add_argument_group("Vespa options")
        g.add_argument(
            "--vespa-url", default="http://localhost",
            help="[Vespa] Base URL without port (default: http://localhost)",
        )
        g.add_argument(
            "--vespa-port", type=int, default=8080,
            help="[Vespa] Query/feed port (default: 8080)",
        )
        g.add_argument(
            "--vespa-config-url", default="",
            help="[Vespa] Config server URL (default: same as --vespa-url)",
        )
        g.add_argument(
            "--vespa-config-port", type=int, default=19071,
            help="[Vespa] Config server port (default: 19071)",
        )
        g.add_argument(
            "--vespa-query-mode", default="rrf",
            choices=["rrf", "native_rrf", "hybrid", "dense", "sparse",
                     "bm25_text", "rrf_bm25_text", "hybrid_bm25_text"],
            help=(
                "[Vespa] Query mode: "
                "rrf = two-query Python RRF (comparable to Qdrant RRF, best recall); "
                "native_rrf = Vespa server-side RRF with pre-computed sparse; "
                "hybrid = linear combination alpha*dense + (1-alpha)*sparse; "
                "dense = ANN only; sparse = full-scan dot-product; "
                "bm25_text = Vespa native BM25 on raw text (requires --hf-dataset-id); "
                "rrf_bm25_text = RRF(closeness, bm25) on ANN+text candidates; "
                "hybrid_bm25_text = alpha*closeness + (1-alpha)*bm25(text) "
                "(default: rrf)"
            ),
        )
        g.add_argument(
            "--vespa-rrf-k", type=int, default=60,
            help="[Vespa] RRF constant k in 1/(k+rank). Only used in rrf mode (default: 60)",
        )
        g.add_argument(
            "--vespa-target-hits-multiplier", type=int, default=10,
            help="[Vespa] Candidate pool = top_k * multiplier for dense/rrf queries (default: 10)",
        )
        g.add_argument(
            "--vespa-alpha", type=float, default=0.5,
            help="[Vespa] Dense weight for hybrid mode: alpha*dense + (1-alpha)*sparse (default: 0.5)",
        )
        g.add_argument(
            "--vespa-ef-construction", type=int, default=128,
            help="[Vespa] HNSW ef_construction: neighbors explored at insert time (default: 128)",
        )
        g.add_argument(
            "--vespa-ef-search", type=int, default=128,
            help="[Vespa] HNSW ef_search: min candidates explored at query time (default: 128)",
        )
        g.add_argument(
            "--vespa-precision", default="float32",
            choices=["float32", "bfloat16", "int8"],
            help="[Vespa] Vector precision: float32 (4B), bfloat16 (2B), int8 (1B) (default: float32)",
        )
        g.add_argument(
            "--vespa-rerank-count", type=int, default=1000,
            help="[Vespa] GlobalPhaseRanking rerank_count: candidates reranked server-side (default: 1000)",
        )

    @staticmethod
    def build_config(args) -> dict:
        return {
            "url":                    args.vespa_url,
            "port":                   args.vespa_port,
            "config_url":             args.vespa_config_url,
            "config_port":            args.vespa_config_port,
            "query_mode":             args.vespa_query_mode,
            "rrf_k":                  args.vespa_rrf_k,
            "target_hits_multiplier": args.vespa_target_hits_multiplier,
            "alpha":                  args.vespa_alpha,
            "ef_construction":        args.vespa_ef_construction,
            "ef_search":              args.vespa_ef_search,
            "precision":              args.vespa_precision,
            "rerank_count":           args.vespa_rerank_count,
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _scale_vector(vector: List[float], precision: str) -> List[float]:
    """
    Scale a float32 vector for int8 storage/query.

    int8 range is [-128, 127]. Standard embeddings are in [-1, 1].
    Multiply by 127 to map [-1, 1] → [-127, 127].
    For float32 and bfloat16, no scaling needed — returns vector unchanged.
    """
    if precision == "int8":
        return [float(v * 127) for v in vector]
    return vector


def _sparse_cells(indices: List[int], values: List[float]) -> dict:
    """
    Format a SPLADE/BM25 sparse vector as Vespa's mapped-tensor JSON format.

    Vespa tensor<float>(x{}) expects:
        {"cells": {"dimension_index_as_string": float_value, ...}}
    """
    return {"cells": {str(i): float(v) for i, v in zip(indices, values)}}


def _rrf_fuse(
    dense_hits: List[Dict],
    sparse_hits: List[Dict],
    k: int,
    top_k: int,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion matching Qdrant's FusionQuery(RRF).

        score(doc) = 1/(k + dense_rank) + 1/(k + sparse_rank)

    Documents absent from a list get rank = len(list), contributing
    minimally (same as Qdrant's unranked-document treatment).
    """
    pool_size = max(len(dense_hits), len(sparse_hits))

    dense_rank = {h["id"]: i for i, h in enumerate(dense_hits)}
    sparse_rank = {h["id"]: i for i, h in enumerate(sparse_hits)}

    all_ids = set(dense_rank) | set(sparse_rank)
    scores: Dict[str, float] = {}
    for doc_id in all_ids:
        dr = dense_rank.get(doc_id, pool_size)
        sr = sparse_rank.get(doc_id, pool_size)
        scores[doc_id] = 1.0 / (k + dr) + 1.0 / (k + sr)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": doc_id, "score": score} for doc_id, score in ranked[:top_k]]



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import datetime
# import logging
# import time
# from typing import Dict, List, Optional

# import requests
# from vespa.application import Vespa
# from vespa.package import (
#     HNSW,
#     ApplicationPackage,
#     Document,
#     Field,
#     Function,
#     GlobalPhaseRanking,
#     RankProfile,
#     Schema,
#     Validation,
#     ValidationID,
# )

# from src.interface import HybridDB

# logger = logging.getLogger(__name__)
# MAX_RETRIES = 10

# _DISTANCE_MAP = {
#     "cosine": "angular",
#     "dot":    "innerproduct",
#     "l2":     "euclidean",
# }


# class VespaDB(HybridDB):
#     """
#     Vespa implementation of HybridDB.

#     Query modes
#     -----------
#     rrf (default) ← BEST RECALL
#         Two queries (dense ANN + sparse full-scan), fused in Python with
#         Reciprocal Rank Fusion: score = 1/(k+dense_rank) + 1/(k+sparse_rank).
#         Comparable to Qdrant's FusionQuery(RRF). Use rrf_k=60 (standard).

#         Key levers for recall:
#           - target_hits_multiplier: candidate_k = top_k * multiplier
#             Higher = better recall, higher latency. Default 20 (was 10).
#           - rrf_k=60: standard constant, do not change for fair comparison.

#         NOTE: sparse query is a full corpus scan — catches sparse-only relevant
#         docs that ANN misses. This is why recall > native_rrf.

#     native_rrf
#         Single dense-ANN query. Vespa computes RRF server-side in global phase.
#         Faster (one round-trip) but lower recall — only sees ANN candidates,
#         sparse-only relevant docs are missed.

#     hybrid
#         Single dense-ANN query. Linear combination:
#             alpha * closeness + (1 - alpha) * sparse_dot_product
#         Fast but scale mismatch between dense and sparse scores.

#     dense
#         Pure dense ANN retrieval only. Recall baseline.

#     sparse
#         Pure sparse full-scan scored by dot-product. Recall baseline.

#     Apple-to-apple comparison with Qdrant/Endee
#     --------------------------------------------
#     Run both with the SAME --sparse-mode (e.g. endee_bm25 or splade).
#     The framework pre-computes sparse vectors and saves them to .npy files;
#     both DBs receive identical dense + sparse input vectors.

#     Example (best recall):
#         python -m src.main --db vespa --vespa-query-mode rrf --vespa-rrf-k 60 \\
#             --vespa-target-hits-multiplier 20 \\
#             --sparse-mode endee_bm25 --index-name scifact_bench ...
#     """

#     def __init__(
#         self,
#         url: str = "http://localhost",
#         port: int = 8080,
#         config_port: int = 19071,
#         query_mode: str = "rrf",
#         target_hits_multiplier: int = 20,   # increased from 10 → better recall
#         rrf_k: int = 60,
#         alpha: float = 0.5,
#     ):
#         self.url = url
#         self.port = port
#         self.config_port = config_port
#         self.query_mode = query_mode
#         self.target_hits_multiplier = target_hits_multiplier
#         self.rrf_k = rrf_k
#         self.alpha = alpha
#         self.client: Optional[Vespa] = None
#         self.schema_name: Optional[str] = None
#         self.dimension: Optional[int] = None
#         self.distance_metric: str = "angular"

#     # ------------------------------------------------------------------
#     # HybridDB interface
#     # ------------------------------------------------------------------

#     def init(
#         self,
#         index_name: str,
#         dimension: int = 384,
#         space_type: str = "cosine",
#         create: bool = True,
#         **kwargs,
#     ) -> None:
#         self.schema_name = index_name
#         self.dimension = dimension
#         self.distance_metric = _DISTANCE_MAP.get(space_type.lower(), "angular")

#         if create:
#             app_package = self._build_application_package()
#             self._deploy(app_package)

#         self.client = Vespa(self.url, port=self.port)
#         self.client.wait_for_application_up(max_wait=300)
#         logger.info(
#             "VespaDB ready at %s:%d (schema=%s, mode=%s, multiplier=%d)",
#             self.url, self.port, self.schema_name,
#             self.query_mode, self.target_hits_multiplier,
#         )

#     def index_batch(self, points: List[Dict]) -> None:
#         docs = []
#         for p in points:
#             if not p["sparse_indices"]:
#                 continue
#             fields = {
#                 "doc_id": str(p["id"]),
#                 "embedding": p["vector"],
#                 "sparse_embedding": _sparse_cells(
#                     p["sparse_indices"], p["sparse_values"]
#                 ),
#             }
#             if p.get("text"):
#                 fields["text"] = p["text"]
#             docs.append({"id": str(p["id"]), "fields": fields})

#         if not docs:
#             return

#         for attempt in range(MAX_RETRIES):
#             try:
#                 failures: List[str] = []

#                 def _cb(response, doc_id):
#                     if not response.is_successful():
#                         failures.append(doc_id)
#                         logger.warning(
#                             "Feed failed for doc %s: %s", doc_id, response.json
#                         )

#                 self.client.feed_iterable(
#                     iter(docs),
#                     schema=self.schema_name,
#                     namespace="hybrid",
#                     callback=_cb,
#                     max_workers=16,
#                     max_connections=16,
#                 )
#                 if failures:
#                     raise RuntimeError(
#                         f"Feed failed for {len(failures)} docs: {failures[:3]}"
#                     )
#                 return
#             except Exception as e:
#                 logger.warning(
#                     "index_batch attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, e
#                 )
#                 if attempt < MAX_RETRIES - 1:
#                     time.sleep(1.5)
#                 else:
#                     logger.error(
#                         "index_batch failed after %d retries (%d points): %s",
#                         MAX_RETRIES, len(points), e,
#                     )
#                     raise

#     def search(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#         text: str = "",
#     ) -> List[Dict]:
#         try:
#             if self.query_mode == "rrf":
#                 return self._search_rrf(
#                     dense_vector, sparse_indices, sparse_values, top_k
#                 )
#             if self.query_mode == "native_rrf":
#                 return self._search_native_rrf(
#                     dense_vector, sparse_indices, sparse_values, top_k
#                 )
#             if self.query_mode in ("bm25_text", "rrf_bm25_text", "hybrid_bm25_text"):
#                 return self._search_text(dense_vector, top_k, text)
#             body = self._build_single_query(
#                 dense_vector, sparse_indices, sparse_values, top_k
#             )
#             result = self.client.query(body)
#             children = result.get_json().get("root", {}).get("children", [])
#             return [
#                 {"id": child["fields"]["doc_id"], "score": child.get("relevance", 0.0)}
#                 for child in children
#             ]
#         except Exception as e:
#             logger.error("search failed: %s", e)
#             raise

#     def list_indices(self) -> List[str]:
#         if self.schema_name:
#             return [self.schema_name]
#         return []

#     # ------------------------------------------------------------------
#     # RRF search — BEST RECALL
#     # Two queries + Python-level fusion
#     # ------------------------------------------------------------------

#     def _search_rrf(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#     ) -> List[Dict]:
#         """
#         Reciprocal Rank Fusion — highest recall mode.

#         Formula: score = 1/(rrf_k + dense_rank) + 1/(rrf_k + sparse_rank)

#         Why best recall:
#           1. Dense leg: ANN retrieves candidate_k = top_k * multiplier docs
#           2. Sparse leg: FULL CORPUS SCAN — every doc scored, none missed
#           3. Union of both candidate sets fused in Python

#         candidate_k = top_k * target_hits_multiplier (default: top_k * 20)
#         Higher multiplier = better recall, higher latency.
#         """
#         candidate_k = top_k * self.target_hits_multiplier

#         # --- dense query: ANN top candidate_k ---
#         dense_body = {
#             "yql": (
#                 f"select doc_id from {self.schema_name} where "
#                 f"{{targetHits: {candidate_k}}}"
#                 f"nearestNeighbor(embedding, q_embedding)"
#             ),
#             "hits": candidate_k,
#             "input.query(q_embedding)": dense_vector,
#             "ranking": "dense",
#         }
#         dense_hits = self._run_query(dense_body)

#         # --- sparse query: full corpus scan, scored by dot product ---
#         # "where true" = scan all docs — this is what gives us superior recall
#         # vs native_rrf which only sees ANN candidates
#         sparse_body = {
#             "yql": f"select doc_id from {self.schema_name} where true",
#             "hits": candidate_k,
#             "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#             "ranking": "sparse",
#         }
#         sparse_hits = self._run_query(sparse_body)

#         # --- Python RRF fusion ---
#         return _rrf_fuse(dense_hits, sparse_hits, k=self.rrf_k, top_k=top_k)

#     # ------------------------------------------------------------------
#     # Native RRF (Vespa server-side, single query)
#     # ------------------------------------------------------------------

#     def _search_native_rrf(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#     ) -> List[Dict]:
#         """
#         Vespa-native RRF via GlobalPhaseRanking with reciprocal_rank_fusion().
#         Single ANN query — faster than Python RRF but lower recall because
#         sparse-only relevant docs (not in ANN candidates) are missed.
#         """
#         candidate_k = top_k * self.target_hits_multiplier
#         body = {
#             "yql": (
#                 f"select doc_id from {self.schema_name} where "
#                 f"{{targetHits: {candidate_k}}}"
#                 f"nearestNeighbor(embedding, q_embedding)"
#             ),
#             "hits": top_k,
#             "input.query(q_embedding)": dense_vector,
#             "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#             "ranking": "native_rrf",
#         }
#         return self._run_query(body)

#     # ------------------------------------------------------------------
#     # Native BM25 text modes
#     # ------------------------------------------------------------------

#     def _search_text(
#         self,
#         dense_vector: List[float],
#         top_k: int,
#         text: str,
#     ) -> List[Dict]:
#         candidate_k = top_k * self.target_hits_multiplier

#         if self.query_mode == "bm25_text":
#             body = {
#                 "yql": f"select doc_id from {self.schema_name} where userQuery()",
#                 "query": text,
#                 "hits": top_k,
#                 "ranking": "bm25_text",
#             }
#         elif self.query_mode == "rrf_bm25_text":
#             body = {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {candidate_k}}}nearestNeighbor(embedding, q_embedding)"
#                     f" OR userQuery()"
#                 ),
#                 "query": text,
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "ranking": "rrf_bm25_text",
#             }
#         else:  # hybrid_bm25_text
#             body = {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {candidate_k}}}nearestNeighbor(embedding, q_embedding)"
#                     f" OR userQuery()"
#                 ),
#                 "query": text,
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "input.query(alpha)": self.alpha,
#                 "ranking": "hybrid_bm25_text",
#             }
#         return self._run_query(body)

#     def _run_query(self, body: dict) -> List[Dict]:
#         result = self.client.query(body)
#         children = result.get_json().get("root", {}).get("children", [])
#         return [
#             {"id": child["fields"]["doc_id"], "score": child.get("relevance", 0.0)}
#             for child in children
#         ]

#     # ------------------------------------------------------------------
#     # Single-query modes (hybrid / dense / sparse)
#     # ------------------------------------------------------------------

#     def _build_single_query(
#         self,
#         dense_vector: List[float],
#         sparse_indices: List[int],
#         sparse_values: List[float],
#         top_k: int,
#     ) -> dict:
#         candidate_hits = top_k * self.target_hits_multiplier

#         if self.query_mode == "hybrid":
#             return {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {candidate_hits}}}"
#                     f"nearestNeighbor(embedding, q_embedding)"
#                 ),
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#                 "input.query(alpha)": self.alpha,
#                 "ranking": "hybrid",
#             }

#         if self.query_mode == "dense":
#             return {
#                 "yql": (
#                     f"select doc_id from {self.schema_name} where "
#                     f"{{targetHits: {top_k}}}"
#                     f"nearestNeighbor(embedding, q_embedding)"
#                 ),
#                 "hits": top_k,
#                 "input.query(q_embedding)": dense_vector,
#                 "ranking": "dense",
#             }

#         # sparse — full-scan scored by dot product
#         return {
#             "yql": f"select doc_id from {self.schema_name} where true",
#             "hits": top_k,
#             "input.query(q_sparse)": _sparse_cells(sparse_indices, sparse_values),
#             "ranking": "sparse",
#         }

#     # ------------------------------------------------------------------
#     # Application package
#     # ------------------------------------------------------------------

#     def _build_application_package(self) -> ApplicationPackage:
#         fields = [
#             Field(
#                 name="doc_id",
#                 type="string",
#                 indexing=["summary", "attribute"],
#             ),
#             Field(
#                 name="embedding",
#                 type=f"tensor<float>(x[{self.dimension}])",
#                 indexing=["summary", "attribute", "index"],
#                 ann=HNSW(
#                     distance_metric=self.distance_metric,
#                     max_links_per_node=16,
#                     neighbors_to_explore_at_insert=200,
#                 ),
#             ),
#             Field(
#                 name="sparse_embedding",
#                 type="tensor<float>(x{})",
#                 indexing=["summary", "attribute"],
#             ),
#             Field(
#                 name="text",
#                 type="string",
#                 indexing=["index", "summary"],
#                 index="enable-bm25",
#             ),
#         ]

#         rank_profiles = [
#             # Dense ANN — used by rrf (dense leg), native_rrf, hybrid
#             RankProfile(
#                 name="dense",
#                 first_phase="closeness(field, embedding)",
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                 ],
#             ),
#             # Sparse dot-product — used by rrf (sparse leg, full scan) and standalone sparse
#             RankProfile(
#                 name="sparse",
#                 first_phase="sum(query(q_sparse) * attribute(sparse_embedding))",
#                 inputs=[
#                     ("query(q_sparse)", "tensor<float>(x{})"),
#                 ],
#             ),
#             # Linear hybrid (alpha * dense + (1-alpha) * sparse)
#             RankProfile(
#                 name="hybrid",
#                 first_phase=(
#                     "query(alpha) * closeness(field, embedding) + "
#                     "(1 - query(alpha)) * sum(query(q_sparse) * attribute(sparse_embedding))"
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                     ("query(q_sparse)", "tensor<float>(x{})"),
#                     ("query(alpha)", "double"),
#                 ],
#             ),
#             # Pure BM25 text search
#             RankProfile(
#                 name="bm25_text",
#                 first_phase="bm25(text)",
#             ),
#             # Native RRF: ANN OR text, global-phase RRF(closeness, bm25)
#             RankProfile(
#                 name="rrf_bm25_text",
#                 first_phase="closeness(field, embedding) + bm25(text)",
#                 global_phase=GlobalPhaseRanking(
#                     expression="reciprocal_rank_fusion(closeness(field, embedding), bm25(text))",
#                     rerank_count=1000,
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                 ],
#             ),
#             # Linear hybrid: alpha*closeness + (1-alpha)*bm25(text)
#             RankProfile(
#                 name="hybrid_bm25_text",
#                 first_phase=(
#                     "query(alpha) * closeness(field, embedding) + "
#                     "(1 - query(alpha)) * bm25(text)"
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                     ("query(alpha)", "double"),
#                 ],
#             ),
#             # Vespa-native RRF: single ANN query, server-side global-phase fusion
#             # k=60 by default (matches Python RRF rrf_k=60)
#             RankProfile(
#                 name="native_rrf",
#                 first_phase="closeness(field, embedding)",
#                 functions=[
#                     Function(
#                         name="sparse_score",
#                         expression="sum(query(q_sparse) * attribute(sparse_embedding))",
#                     ),
#                 ],
#                 global_phase=GlobalPhaseRanking(
#                     expression="reciprocal_rank_fusion(closeness(field, embedding), sparse_score)",
#                     rerank_count=1000,
#                 ),
#                 inputs=[
#                     ("query(q_embedding)", f"tensor<float>(x[{self.dimension}])"),
#                     ("query(q_sparse)", "tensor<float>(x{})"),
#                 ],
#             ),
#         ]

#         tomorrow = datetime.date.today() + datetime.timedelta(days=1)

#         return ApplicationPackage(
#             name="hybridbench",
#             schema=[
#                 Schema(
#                     name=self.schema_name,
#                     document=Document(fields=fields),
#                     rank_profiles=rank_profiles,
#                 )
#             ],
#             validations=[
#                 Validation(ValidationID.tensorTypeChange, until=tomorrow),
#                 Validation(ValidationID.fieldTypeChange, until=tomorrow),
#                 Validation(ValidationID.contentClusterRemoval, until=tomorrow),
#             ],
#         )

#     def _deploy(self, app_package: ApplicationPackage) -> None:
#         deploy_url = (
#             f"{self.url}:{self.config_port}"
#             "/application/v2/tenant/default/prepareandactivate"
#         )
#         package_data = app_package.to_zip()
#         response = requests.post(
#             url=deploy_url,
#             data=package_data,
#             headers={"Content-Type": "application/zip"},
#             timeout=120,
#         )
#         if not response.ok:
#             logger.error(
#                 "Deploy failed %d: %s", response.status_code, response.text
#             )
#         response.raise_for_status()
#         logger.info(
#             "Deployed Vespa application 'hybridbench' with schema '%s'",
#             self.schema_name,
#         )

#     # ------------------------------------------------------------------
#     # CLI integration
#     # ------------------------------------------------------------------

#     @staticmethod
#     def add_args(parser) -> None:
#         g = parser.add_argument_group("Vespa options")
#         g.add_argument(
#             "--vespa-url", default="http://localhost",
#             help="[Vespa] Base URL without port (default: http://localhost)",
#         )
#         g.add_argument(
#             "--vespa-port", type=int, default=8080,
#             help="[Vespa] Query/feed port (default: 8080)",
#         )
#         g.add_argument(
#             "--vespa-config-port", type=int, default=19071,
#             help="[Vespa] Config server port (default: 19071)",
#         )
#         g.add_argument(
#             "--vespa-query-mode", default="rrf",
#             choices=["rrf", "native_rrf", "hybrid", "dense", "sparse",
#                      "bm25_text", "rrf_bm25_text", "hybrid_bm25_text"],
#             help=(
#                 "[Vespa] Query mode. "
#                 "rrf = best recall: two-query Python RRF with full sparse scan; "
#                 "native_rrf = Vespa server-side RRF, single query, lower recall; "
#                 "hybrid = linear alpha*dense + (1-alpha)*sparse; "
#                 "dense = ANN only; sparse = full-scan dot-product; "
#                 "bm25_text = Vespa native BM25 on raw text; "
#                 "rrf_bm25_text = RRF(closeness, bm25) ANN+text candidates; "
#                 "hybrid_bm25_text = alpha*closeness + (1-alpha)*bm25(text) "
#                 "(default: rrf)"
#             ),
#         )
#         g.add_argument(
#             "--vespa-rrf-k", type=int, default=60,
#             help="[Vespa] RRF k constant: 1/(k+rank). Keep at 60 for fair comparison (default: 60)",
#         )
#         g.add_argument(
#             "--vespa-target-hits-multiplier", type=int, default=20,
#             help=(
#                 "[Vespa] candidate_k = top_k * multiplier. "
#                 "Higher = better recall, higher latency. "
#                 "Default 20 (use 10 for speed, 50 for max recall)"
#             ),
#         )
#         g.add_argument(
#             "--vespa-alpha", type=float, default=0.5,
#             help="[Vespa] Dense weight for hybrid modes: alpha*dense + (1-alpha)*sparse (default: 0.5)",
#         )

#     @staticmethod
#     def build_config(args) -> dict:
#         return {
#             "url":                    args.vespa_url,
#             "port":                   args.vespa_port,
#             "config_port":            args.vespa_config_port,
#             "query_mode":             args.vespa_query_mode,
#             "rrf_k":                  args.vespa_rrf_k,
#             "target_hits_multiplier": args.vespa_target_hits_multiplier,
#             "alpha":                  args.vespa_alpha,
#         }


# # ------------------------------------------------------------------
# # Module-level helpers
# # ------------------------------------------------------------------

# def _sparse_cells(indices: List[int], values: List[float]) -> dict:
#     """
#     Format a SPLADE/BM25 sparse vector as Vespa's mapped-tensor JSON format.
#     Vespa tensor<float>(x{}) expects:
#         {"cells": {"dimension_index_as_string": float_value, ...}}
#     """
#     return {"cells": {str(i): float(v) for i, v in zip(indices, values)}}


# def _rrf_fuse(
#     dense_hits: List[Dict],
#     sparse_hits: List[Dict],
#     k: int,
#     top_k: int,
# ) -> List[Dict]:
#     """
#     Reciprocal Rank Fusion matching Qdrant's FusionQuery(RRF).

#         score(doc) = 1/(k + dense_rank) + 1/(k + sparse_rank)

#     k=60 (standard, matches Qdrant and Vespa native default).
#     Docs absent from a list get rank = pool_size (minimal contribution).
#     """
#     pool_size = max(len(dense_hits), len(sparse_hits))

#     dense_rank  = {h["id"]: i for i, h in enumerate(dense_hits)}
#     sparse_rank = {h["id"]: i for i, h in enumerate(sparse_hits)}

#     all_ids = set(dense_rank) | set(sparse_rank)
#     scores: Dict[str, float] = {}
#     for doc_id in all_ids:
#         dr = dense_rank.get(doc_id, pool_size)
#         sr = sparse_rank.get(doc_id, pool_size)
#         scores[doc_id] = 1.0 / (k + dr) + 1.0 / (k + sr)

#     ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     return [{"id": doc_id, "score": score} for doc_id, score in ranked[:top_k]]