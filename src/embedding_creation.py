import argparse
import gc
import logging
import numpy.lib.format as nf
from pathlib import Path

import mmh3
import numpy as np
from datasets import load_dataset
from endee_model import SparseModel
from endee_model.sparse.bm25 import Bm25
from rank_bm25 import BM25L
from sentence_transformers import SentenceTransformer, SparseEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DENSE_MODEL_ID     = "sentence-transformers/all-MiniLM-L6-v2"
SPLADE_MODEL_ID    = "prithivida/Splade_PP_en_v1"
ENDEE_BM25_MODEL_ID = "endee/bm25"
DATA_DIR           = Path("data")

# ── rank_bm25 helpers ─────────────────────────────────────────────────────────
_bm25_tokenizer = Bm25(ENDEE_BM25_MODEL_ID)

def _tokenize(text: str) -> list[str]:
    tokens = _bm25_tokenizer._tokenizer.tokenize(text.lower())
    return _bm25_tokenizer._normalize(tokens)

def _token_id(token: str) -> int:
    return abs(mmh3.hash(token))

def _sparse_vector_bm25l(bm25: BM25L, doc_idx: int) -> tuple[list[int], list[float]]:
    """Extract BM25L TF*IDF sparse vector for a single document."""
    doc_freqs = bm25.doc_freqs[doc_idx]
    doc_len   = bm25.doc_len[doc_idx]
    indices: list[int]   = []
    values:  list[float] = []
    for term, freq in doc_freqs.items():
        idf = bm25.idf.get(term, 0.0)
        if idf <= 0.0:
            continue
        ctf = freq / (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)
        tf  = (bm25.k1 + 1) * (ctf + bm25.delta) / (bm25.k1 + ctf + bm25.delta)
        indices.append(_token_id(term))
        values.append(float(idf * tf))
    return indices, values

DATASET_CONFIG = {
    "beir_scifact": {"hf_name": "BeIR/scifact-qrels", "split": "train"},
    "beir_quora":   {"hf_name": "BeIR/quora-qrels",   "split": "test"},
    "beir_msmarco": {"hf_name": "BeIR/msmarco-qrels", "split": "dev"},
}


def extract_dataset_name(dataset_id: str) -> str:
    """'BeIR/scifact' → 'beir_scifact'"""
    return dataset_id.replace("/", "_").lower()


def load_hf_dataset(dataset_id: str, split: str, cache_dir: str = None):
    """
    Load a HuggingFace BeIR-style dataset.
    split: 'corpus' or 'queries'
    """
    logger.info("Loading dataset '%s' config='%s' split='%s'", dataset_id, split, split)
    dataset = load_dataset(dataset_id, split, split=split, cache_dir=cache_dir)
    logger.info("Loaded %d records", len(dataset))
    return dataset

def create_dense_embeddings(
    dataset,
    dataset_name: str,
    split: str,
    batch_size: int = 1000,
    device: str = "cpu",
    dense_model_id: str = DENSE_MODEL_ID,
    cache_dir: str = None,
    workers: int = 4,
):
    """
    Writes embeddings directly into the final .npy file via memmap — no temp files, no full RAM load.
    Peak RAM = one batch of embeddings only.

    The .npy header is written first (shape + dtype), then the data region is memory-mapped
    and filled batch by batch. Result is a valid .npy loadable with mmap_mode='r'.

    Output:
      data/<dataset_name>_dense_<split>.npy      — float32 (N, dim), memory-mappable
      data/<dataset_name>_dense_<split>_ids.npy  — str (N,)
    """
    logger.info("Loading dense model: %s", dense_model_id)
    model = SentenceTransformer(dense_model_id, device=device, cache_folder=cache_dir)

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)
    dim     = model.get_sentence_embedding_dimension()

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{dataset_name}_dense_{split}.npy"
    ids_path = out_dir / f"{dataset_name}_dense_{split}_ids.npy"

    # Write .npy header then pre-extend file to full size (all zeros)
    # This makes npy_path a valid .npy from the start — no temp file needed
    with open(npy_path, "wb") as f:
        nf.write_array_header_1_0(f, {
            "descr": "<f8",
            "fortran_order": False,
            "shape": (n_docs, dim),
        })
        header_size = f.tell()
        f.seek(header_size + n_docs * dim * 8 - 1)
        f.write(b"\x00")

    # Memory-map the data region and fill batch by batch
    mmap = np.memmap(npy_path, dtype="float64", mode="r+", offset=header_size, shape=(n_docs, dim))

    pool = model.start_multi_process_pool(target_devices=["cpu"] * workers)
    logger.info("Encoding %d texts (dim=%d) using %d workers ...", n_docs, dim, workers)

    for i in tqdm(range(0, n_docs, batch_size), desc=f"Dense {split}"):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(batch, pool=pool, show_progress_bar=False)
        mmap[i : i + len(batch)] = vecs
        mmap.flush()
        del vecs
        gc.collect()

    model.stop_multi_process_pool(pool)
    del mmap

    np.save(ids_path, np.array(doc_ids))

    logger.info("Saved dense (%d × %d) → %s", n_docs, dim, npy_path)
    logger.info("Saved IDs → %s", ids_path)
    return (n_docs, dim)


def _stream_sparse_to_npy(base_path: Path, doc_ids: list, encode_iter):
    """
    Streams (indices, values) one doc at a time directly into temp binary files,
    then writes final .npy files with proper headers. Peak RAM = one batch at a time.

    Why temp files?
      np.save / .npy format requires the total array shape in the file header, which
      must be written BEFORE any data. We don't know total_nnz until all docs are
      processed. So we first stream raw bytes into .tmp files (no header needed),
      then once total_nnz is known we write the .npy header and chunk-copy the data.
      This way the full values/indices arrays are never loaded into RAM simultaneously.

    Output files (base_path = e.g. data/scifact_sparse_corpus_bm25):
      <base_path>_values.npy       — float32 (total_nnz,)
      <base_path>_col_indices.npy  — int32   (total_nnz,)
      <base_path>_indptr.npy       — int64   (n_docs+1,)  start offset of each doc in flat arrays
      <base_path>_ids.npy          — str     (n_docs,)

    Load:
      values  = np.load("..._values.npy")
      indices = np.load("..._col_indices.npy")
      indptr  = np.load("..._indptr.npy")
      # doc i: values[indptr[i]:indptr[i+1]], indices[indptr[i]:indptr[i+1]]
    """
    tmp_val = str(base_path) + ".tmp_val"
    tmp_idx = str(base_path) + ".tmp_idx"
    indptr    = [0]
    kept_ids  = []
    skipped   = 0

    # Pass 1: write each doc's bytes to disk immediately — nothing accumulates in RAM
    # Docs with empty sparse vectors are skipped entirely (not stored, not in ids)
    with open(tmp_val, "wb") as fv, open(tmp_idx, "wb") as fi:
        for (indices, values), doc_id in zip(encode_iter, doc_ids):
            if len(indices) == 0:
                skipped += 1
                continue
            np.array(values,  dtype=np.float64).tofile(fv)
            np.array(indices, dtype=np.int32).tofile(fi)
            indptr.append(indptr[-1] + len(indices))
            kept_ids.append(doc_id)

    if skipped:
        logger.info("Skipped %d docs with empty sparse vectors", skipped)

    total_nnz = indptr[-1]

    # Pass 2: prepend .npy header then chunk-copy from temp — still no full RAM load
    for path, tmp, descr in [
        (str(base_path) + "_values.npy",     tmp_val, "<f8"),
        (str(base_path) + "_col_indices.npy", tmp_idx, "<i4"),
    ]:
        with open(path, "wb") as dst:
            nf.write_array_header_1_0(dst, {"descr": descr, "fortran_order": False, "shape": (total_nnz,)})
            with open(tmp, "rb") as src:
                while True:
                    chunk = src.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

    Path(tmp_val).unlink()
    Path(tmp_idx).unlink()

    np.save(str(base_path) + "_indptr.npy", np.array(indptr,   dtype=np.int64))
    np.save(str(base_path) + "_ids.npy",    np.array(kept_ids))

    logger.info("Saved %d sparse vectors (%d nnz) → %s_*.npy", len(kept_ids), total_nnz, base_path)


def create_sparse_embeddings_endee_bm25(
    dataset,
    dataset_name: str,
    split: str,
    batch_size: int = 1000,
):
    """
    Encode with Endee BM25 model batch by batch (Endee-specific, requires Endee server).
      corpus  → model.embed()
      queries → model.query_embed()

    Output: data/<dataset_name>_sparse_<split>_endee_bm25_*.npy
    """
    logger.info("Loading Endee BM25 model: %s", ENDEE_BM25_MODEL_ID)
    model    = SparseModel(model_name=ENDEE_BM25_MODEL_ID)
    embed_fn = model.embed if split == "corpus" else model.query_embed

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    def _endee_bm25_iter():
        for i in tqdm(range(0, n_docs, batch_size), desc=f"Endee BM25 {split}"):
            batch_vecs = list(embed_fn(texts[i : i + batch_size]))
            for sv in batch_vecs:
                yield sv.indices.tolist(), sv.values.tolist()
            del batch_vecs
            gc.collect()

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_endee_bm25"
    _stream_sparse_to_npy(base_path, doc_ids, _endee_bm25_iter())
    return n_docs


def create_sparse_embeddings_bm25(
    dataset,
    dataset_name: str,
    split: str,
):
    """
    Encode with rank_bm25 (BM25L). Builds IDF over the full split upfront,
    then extracts sparse TF*IDF vectors. Works with any DB.

    Output: data/<dataset_name>_sparse_<split>_bm25_*.npy
    """
    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    logger.info("Tokenizing %d documents for BM25L index...", n_docs)
    tokenized = [_tokenize(t) for t in tqdm(texts, desc=f"Tokenizing {split}")]

    logger.info("Building BM25L index...")
    bm25 = BM25L(tokenized)
    del tokenized
    gc.collect()

    def _bm25_iter():
        for i in range(n_docs):
            yield _sparse_vector_bm25l(bm25, i)

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_bm25"
    _stream_sparse_to_npy(base_path, doc_ids, _bm25_iter())
    return n_docs


def create_sparse_embeddings_pymilvus_bm25(
    dataset,
    dataset_name: str,
    split: str,
    batch_size: int = 1000,
):
    """
    Encode with PyMilvus BM25EmbeddingFunction (client-side BM25 from pymilvus.model.sparse).
    Corpus pass: fits the BM25 model and saves it to disk for query encoding.
    Query pass:  loads the saved model and encodes queries.

    Output: data/<dataset_name>_sparse_<split>_pymilvus_bm25_*.npy
    """
    from pymilvus.model.sparse import BM25EmbeddingFunction

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    out_dir    = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(out_dir / f"{dataset_name}_pymilvus_bm25_model.json")

    ef = BM25EmbeddingFunction()

    if split == "corpus":
        logger.info("Fitting PyMilvus BM25 on %d corpus documents...", n_docs)
        ef.fit(texts)
        ef.save(model_path)
        logger.info("Saved PyMilvus BM25 model → %s", model_path)
        encode_fn = ef.encode_documents
    else:
        logger.info("Loading PyMilvus BM25 model from %s", model_path)
        ef.load(model_path)
        encode_fn = ef.encode_queries

    def _pymilvus_bm25_iter():
        for i in tqdm(range(0, n_docs, batch_size), desc=f"PyMilvus BM25 {split}"):
            batch = texts[i : i + batch_size]
            vecs  = encode_fn(batch)
            for j in range(len(batch)):
                row = vecs[j].tocsr()
                yield row.indices.tolist(), row.data.tolist()
            del vecs
            gc.collect()

    base_path = out_dir / f"{dataset_name}_sparse_{split}_pymilvus_bm25"
    _stream_sparse_to_npy(base_path, doc_ids, _pymilvus_bm25_iter())
    return n_docs


def create_sparse_embeddings_milvus_splade(
    dataset,
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    device: str = "cpu",
    milvus_splade_model: str = "naver/splade-cocondenser-selfdistil",
):
    """
    Encode with pymilvus SpladeEmbeddingFunction (naver/splade-cocondenser-* family).

    Output: data/<dataset_name>_sparse_<split>_milvus_splade_*.npy
    """
    from pymilvus.model.sparse import SpladeEmbeddingFunction

    logger.info("Loading Milvus SPLADE model: %s", milvus_splade_model)
    ef = SpladeEmbeddingFunction(model_name=milvus_splade_model, device=device)

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    encode_fn = ef.encode_documents if split == "corpus" else ef.encode_queries

    def _milvus_splade_iter():
        for i in tqdm(range(0, n_docs, batch_size), desc=f"Milvus SPLADE {split}"):
            batch = texts[i : i + batch_size]
            vecs  = encode_fn(batch)
            for j in range(len(batch)):
                row = vecs[j].tocsr()
                yield row.indices.tolist(), row.data.tolist()
            del vecs
            gc.collect()

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_milvus_splade"
    _stream_sparse_to_npy(base_path, doc_ids, _milvus_splade_iter())
    return n_docs


def create_sparse_embeddings_splade(
    dataset,
    dataset_name: str,
    split: str,
    batch_size: int = 1000,
    device: str = "cpu",
    sparse_model_id: str = SPLADE_MODEL_ID,
    cache_dir: str = None,
    workers: int = 4,
):
    """
    Encode with SPLADE batch by batch using multi-process pool.
    Peak RAM = one batch of sparse tensors at a time.

    Output: data/<dataset_name>_sparse_<split>_splade.npz
    """
    logger.info("Loading SPLADE model: %s", sparse_model_id)
    model = SparseEncoder(sparse_model_id, device=device, cache_folder=cache_dir)

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    pool = model.start_multi_process_pool(target_devices=["cpu"] * workers)
    logger.info("Encoding %d texts (SPLADE) using %d workers …", n_docs, workers)

    def _splade_iter():
        for i in tqdm(range(0, n_docs, batch_size), desc=f"SPLADE {split}"):
            batch_vecs = model.encode(
                texts[i : i + batch_size], pool=pool, show_progress_bar=False
            )
            for sv in batch_vecs:
                sv = sv.coalesce()
                sentence_indices = sv.indices().tolist()
                indices = sentence_indices[1] if len(sentence_indices) == 2 else sentence_indices[0]
                yield indices, sv.values().tolist()
            del batch_vecs
            gc.collect()

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_splade"
    _stream_sparse_to_npy(base_path, doc_ids, _splade_iter())

    model.stop_multi_process_pool(pool)
    return n_docs

def generate_ground_truth(dataset_name: str, cache_dir: str = None):
    """
    Load HF qrels for the dataset, extract unique relevant corpus IDs,
    and save as <dataset_name>_ground_truth_ids.npy in the dataset folder.
    """
    cfg = DATASET_CONFIG.get(dataset_name)
    if cfg is None:
        raise ValueError(
            f"No DATASET_CONFIG entry for '{dataset_name}'. "
            f"Add it to DATASET_CONFIG. Known: {list(DATASET_CONFIG)}"
        )
    logger.info("Loading qrels: %s  split=%s", cfg["hf_name"], cfg["split"])
    qrels = load_dataset(cfg["hf_name"], split=cfg["split"], cache_dir=cache_dir, trust_remote_code=True)
    unique_ids = sorted(set(str(entry["corpus-id"]) for entry in qrels))
    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_ground_truth_ids.npy"
    np.save(str(out_path), np.array(unique_ids))
    logger.info("Saved %d ground-truth corpus IDs → %s", len(unique_ids), out_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create dense and/or sparse embeddings from a BeIR dataset.\n\n"
            "Output files (in data/):\n"
            "  <dataset>_dense_<split>.npy              — float32 (N,dim)\n"
            "  <dataset>_sparse_<split>_bm25_*.npy      — rank_bm25 CSR arrays (any DB)\n"
            "  <dataset>_sparse_<split>_endee_bm25_*.npy — Endee BM25 CSR arrays\n"
            "  <dataset>_sparse_<split>_splade_*.npy    — SPLADE CSR arrays"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-id",   required=True,
                        help="HuggingFace dataset ID, e.g. 'BeIR/scifact' or 'BeIR/quora'")
    parser.add_argument("--batch-size",   type=int, default=1000,
                        help="Encoding batch size (default: 1000)")
    parser.add_argument("--device",       default="cpu",
                        help="Torch device: cpu | cuda | mps (default: cpu)")
    parser.add_argument("--dense-model",  default=DENSE_MODEL_ID,
                        help=f"Dense model ID (default: {DENSE_MODEL_ID})")
    parser.add_argument("--dense",        action="store_true", default=False,
                        help="Generate dense embeddings")
    parser.add_argument(
        "--sparse-mode", default=None, nargs="+",
        choices=["bm25", "endee_bm25", "splade", "pymilvus_bm25", "milvus_splade"],
        help=(
            "Generate sparse embeddings (one or more):\n"
            "  bm25           – rank_bm25 (any DB)\n"
            "  endee_bm25     – Endee BM25 (Endee-specific)\n"
            "  splade         – prithivida/Splade_PP_en_v1\n"
            "  pymilvus_bm25  – PyMilvus BM25EmbeddingFunction (Milvus)\n"
            "  milvus_splade  – PyMilvus SpladeEmbeddingFunction (Milvus)"
        ),
    )
    parser.add_argument("--splade-model", default=SPLADE_MODEL_ID,
                        help=f"SPLADE model ID (default: {SPLADE_MODEL_ID})")
    parser.add_argument("--milvus-splade-model",
                        default="naver/splade-cocondenser-selfdistil",
                        help="Milvus SPLADE model ID (default: naver/splade-cocondenser-selfdistil)")
    parser.add_argument("--workers",      type=int, default=5,
                        help="Parallel CPU workers for dense + SPLADE encoding (default: 4)")
    parser.add_argument("--cache-dir",    default=None,
                        help="Local cache directory for HuggingFace datasets and models")
    args = parser.parse_args()

    if not args.dense and args.sparse_mode is None and not args.ground_truth:
        parser.error("Specify at least one of --dense, --sparse-mode {bm25,endee_bm25,splade}, or --ground-truth")

    sparse_modes = set(args.sparse_mode) if args.sparse_mode else set()
    dataset_name = extract_dataset_name(args.dataset_id)

    if not args.dense and not sparse_modes:
        return

    for split in ["corpus", "queries"]:
        logger.info("=== Processing split: %s ===", split)
        dataset = load_hf_dataset(args.dataset_id, split, cache_dir=args.cache_dir)

        if args.dense:
            shape = create_dense_embeddings(
                dataset=dataset,
                dataset_name=dataset_name,
                split=split,
                batch_size=args.batch_size,
                device=args.device,
                dense_model_id=args.dense_model,
                cache_dir=args.cache_dir,
                workers=args.workers,
            )
            logger.info("Dense done — %s  shape=%s", split, shape)

        if "endee_bm25" in sparse_modes:
            count = create_sparse_embeddings_endee_bm25(
                dataset=dataset,
                dataset_name=dataset_name,
                split=split,
                batch_size=args.batch_size,
            )
            logger.info("Endee BM25 sparse done — %s  %d vectors", split, count)

        if "bm25" in sparse_modes:
            count = create_sparse_embeddings_bm25(
                dataset=dataset,
                dataset_name=dataset_name,
                split=split,
            )
            logger.info("BM25 sparse done — %s  %d vectors", split, count)

        if "splade" in sparse_modes:
            count = create_sparse_embeddings_splade(
                dataset=dataset,
                dataset_name=dataset_name,
                split=split,
                batch_size=args.batch_size,
                device=args.device,
                sparse_model_id=args.splade_model,
                cache_dir=args.cache_dir,
                workers=args.workers,
            )
            logger.info("SPLADE sparse done — %s  %d vectors", split, count)

        if "pymilvus_bm25" in sparse_modes:
            count = create_sparse_embeddings_pymilvus_bm25(
                dataset=dataset,
                dataset_name=dataset_name,
                split=split,
                batch_size=args.batch_size,
            )
            logger.info("PyMilvus BM25 sparse done — %s  %d vectors", split, count)

        if "milvus_splade" in sparse_modes:
            count = create_sparse_embeddings_milvus_splade(
                dataset=dataset,
                dataset_name=dataset_name,
                split=split,
                batch_size=args.batch_size,
                device=args.device,
                milvus_splade_model=args.milvus_splade_model,
            )
            logger.info("Milvus SPLADE sparse done — %s  %d vectors", split, count)


if __name__ == "__main__":
    main()
