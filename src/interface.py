from abc import ABC, abstractmethod
from typing import Dict, List


class HybridDB(ABC):
    """
    Abstract interface every vector DB implementation must satisfy.

    Lifecycle
    ---------
    1. Construct with connection params:   db = EndeeDB(token=..., base_url=...)
    2. Connect / create an index:          db.init(index_name, dimension, ...)
    3. Upsert batches during indexing:     db.index_batch(points)
    4. Query during benchmarking:          db.search(dense, sp_indices, sp_values, top_k)
    5. Inspect available indices:          db.list_indices()
    """

    @abstractmethod
    def init(
        self,
        index_name: str,
        dimension: int = 384,
        space_type: str = "cosine",
        create: bool = True,
        **kwargs,
    ) -> None:
        """
        Connect to an index, optionally creating it first.

        Parameters
        ----------
        index_name : str
        dimension  : int   Dense vector dimension.
        space_type : str   Distance metric (e.g. "cosine").
        create     : bool  If True, create the index before connecting.
                           Set False when the index already exists (query-only runs).
        **kwargs           DB-specific creation options (e.g. sparse_model, precision).
        """

    @abstractmethod
    def index_batch(self, points: List[Dict]) -> None:
        """
        Upsert a single prepared batch into the index.

        Each point must have at minimum:
          { "id": str, "vector": List[float],
            "sparse_indices": List[int], "sparse_values": List[float] }

        The calling framework owns retry logic and timing.
        """

    @abstractmethod
    def search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
        text: str = "",
    ) -> List[Dict]:
        """
        Query the index and return top-k results.

        Parameters
        ----------
        text : str  Optional raw query text. Used by DBs that support native
                    text search (e.g. Vespa BM25). Ignored by others.

        Returns a list of dicts with at least { "id": str, "score": float }.
        """

    @abstractmethod
    def list_indices(self) -> List[str]:
        """Return names of all indices available on this backend."""
