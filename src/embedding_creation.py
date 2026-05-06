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

CHECKPOINT_INTERVAL = 1000  # save sparse checkpoint every N docs

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
    "beir_msmarco": {"hf_name": "BeIR/msmarco-qrels", "split": "validation"},
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


# ── restart helpers ───────────────────────────────────────────────────────────

def _sparse_output_complete(base_path: Path) -> bool:
    return all(
        Path(str(base_path) + s).exists()
        for s in ("_values.npy", "_col_indices.npy", "_indptr.npy", "_ids.npy")
    )


def _load_sparse_checkpoint(base_path: Path):
    """Returns (docs_done, resume_dict) or (0, None) if no checkpoint."""
    ckpt  = Path(str(base_path) + ".ckpt.npz")
    tmp_v = Path(str(base_path) + ".tmp_val")
    tmp_i = Path(str(base_path) + ".tmp_idx")
    if ckpt.exists() and tmp_v.exists() and tmp_i.exists():
        d = np.load(str(ckpt), allow_pickle=True)
        docs_done = int(d["docs_done"])
        return docs_done, {
            "docs_done": docs_done,
            "indptr":    d["indptr"].tolist(),
            "kept_ids":  d["kept_ids"].tolist(),
        }
    return 0, None


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
    Supports restart: resumes from the last completed batch if interrupted.

    Output:
      data/<dataset_name>_dense_<split>.npy      — float64 (N, dim), memory-mappable
      data/<dataset_name>_dense_<split>_ids.npy  — str (N,)
    """
    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path      = out_dir / f"{dataset_name}_dense_{split}.npy"
    ids_path      = out_dir / f"{dataset_name}_dense_{split}_ids.npy"
    progress_path = out_dir / f"{dataset_name}_dense_{split}.npy.progress"

    if ids_path.exists():
        logger.info("Dense %s already complete — skipping", split)
        existing = np.load(str(npy_path), mmap_mode="r")
        return (existing.shape[0], existing.shape[1])

    logger.info("Loading dense model: %s", dense_model_id)
    model = SentenceTransformer(dense_model_id, device=device, cache_folder=cache_dir)
    dim   = model.get_sentence_embedding_dimension()

    start_i = 0
    if progress_path.exists() and npy_path.exists():
        try:
            start_i = int(progress_path.read_text().strip())
            logger.info("Resuming dense %s from doc %d/%d", split, start_i, n_docs)
        except ValueError:
            start_i = 0

    if start_i == 0 or not npy_path.exists():
        with open(npy_path, "wb") as f:
            nf.write_array_header_1_0(f, {
                "descr": "<f8",
                "fortran_order": False,
                "shape": (n_docs, dim),
            })
            header_size = f.tell()
            f.seek(header_size + n_docs * dim * 8 - 1)
            f.write(b"\x00")
    else:
        import io
        buf = io.BytesIO()
        nf.write_array_header_1_0(buf, {"descr": "<f8", "fortran_order": False, "shape": (n_docs, dim)})
        header_size = buf.tell()

    mmap = np.memmap(npy_path, dtype="float64", mode="r+", offset=header_size, shape=(n_docs, dim))
    pool = model.start_multi_process_pool(target_devices=["cpu"] * workers)
    logger.info("Encoding %d texts (dim=%d) using %d workers, starting at doc %d ...", n_docs, dim, workers, start_i)

    for i in tqdm(range(start_i, n_docs, batch_size), desc=f"Dense {split}"):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(batch, pool=pool, show_progress_bar=False)
        mmap[i : i + len(batch)] = vecs
        mmap.flush()
        progress_path.write_text(str(i + batch_size))
        del vecs
        gc.collect()

    model.stop_multi_process_pool(pool)
    del mmap

    np.save(ids_path, np.array(doc_ids))
    progress_path.unlink(missing_ok=True)

    logger.info("Saved dense (%d × %d) → %s", n_docs, dim, npy_path)
    logger.info("Saved IDs → %s", ids_path)
    return (n_docs, dim)


def _stream_sparse_to_npy(base_path: Path, doc_ids: list, encode_iter, _resume=None):
    """
    Streams (indices, values) one doc at a time directly into temp binary files,
    then writes final .npy files with proper headers. Supports restart via _resume.

    _resume = None (fresh start) or dict with keys:
      docs_done  — number of original docs already processed
      indptr     — accumulated indptr list so far
      kept_ids   — accumulated kept_ids list so far
    Tmp files are opened in append mode when resuming.

    Output files (base_path = e.g. data/scifact_sparse_corpus_bm25):
      <base_path>_values.npy       — float64 (total_nnz,)
      <base_path>_col_indices.npy  — int32   (total_nnz,)
      <base_path>_indptr.npy       — int64   (n_docs+1,)
      <base_path>_ids.npy          — str     (n_docs,)
    """
    tmp_val   = str(base_path) + ".tmp_val"
    tmp_idx   = str(base_path) + ".tmp_idx"
    ckpt_path = str(base_path) + ".ckpt.npz"

    if _resume is not None:
        indptr      = _resume["indptr"]
        kept_ids    = _resume["kept_ids"]
        docs_offset = _resume["docs_done"]
        mode        = "ab"
    else:
        indptr      = [0]
        kept_ids    = []
        docs_offset = 0
        mode        = "wb"

    skipped = 0

    with open(tmp_val, mode) as fv, open(tmp_idx, mode) as fi:
        for n, ((indices, values), doc_id) in enumerate(zip(encode_iter, doc_ids)):
            if len(indices) == 0:
                skipped += 1
                continue
            np.array(values,  dtype=np.float64).tofile(fv)
            np.array(indices, dtype=np.int32).tofile(fi)
            indptr.append(indptr[-1] + len(indices))
            kept_ids.append(doc_id)

            if (n + 1) % CHECKPOINT_INTERVAL == 0:
                np.savez(
                    ckpt_path,
                    docs_done=np.array(docs_offset + n + 1, dtype=np.int64),
                    indptr=np.array(indptr, dtype=np.int64),
                    kept_ids=np.array(kept_ids),
                )

    if skipped:
        logger.info("Skipped %d docs with empty sparse vectors", skipped)

    total_nnz = indptr[-1]

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
    Path(ckpt_path).unlink(missing_ok=True)

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
    Supports restart: resumes from last checkpoint if interrupted.

    Output: data/<dataset_name>_sparse_<split>_endee_bm25_*.npy
    """
    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_endee_bm25"

    if _sparse_output_complete(base_path):
        logger.info("Endee BM25 sparse %s already complete — skipping", split)
        return len(dataset["text"])

    docs_done, resume = _load_sparse_checkpoint(base_path)

    texts   = list(dataset["text"])[docs_done:]
    doc_ids = list(dataset["_id"])[docs_done:]
    n_docs  = len(dataset["text"])

    if docs_done > 0:
        logger.info("Resuming Endee BM25 sparse %s from doc %d/%d", split, docs_done, n_docs)

    logger.info("Loading Endee BM25 model: %s", ENDEE_BM25_MODEL_ID)
    model    = SparseModel(model_name=ENDEE_BM25_MODEL_ID)
    embed_fn = model.embed if split == "corpus" else model.query_embed

    def _endee_bm25_iter():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Endee BM25 {split}"):
            batch_vecs = list(embed_fn(texts[i : i + batch_size]))
            for sv in batch_vecs:
                yield sv.indices.tolist(), sv.values.tolist()
            del batch_vecs
            gc.collect()

    _stream_sparse_to_npy(base_path, doc_ids, _endee_bm25_iter(), _resume=resume)
    return n_docs


def create_sparse_embeddings_bm25(
    dataset,
    dataset_name: str,
    split: str,
):
    """
    Encode with rank_bm25 (BM25L). Always builds full IDF index, then resumes
    vector extraction from checkpoint if interrupted.

    Output: data/<dataset_name>_sparse_<split>_bm25_*.npy
    """
    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_bm25"

    if _sparse_output_complete(base_path):
        logger.info("BM25 sparse %s already complete — skipping", split)
        return len(dataset["text"])

    docs_done, resume = _load_sparse_checkpoint(base_path)

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    # Always build the full BM25L index — IDF requires the full corpus
    logger.info("Tokenizing %d documents for BM25L index...", n_docs)
    tokenized = [_tokenize(t) for t in tqdm(texts, desc=f"Tokenizing {split}")]
    logger.info("Building BM25L index...")
    bm25 = BM25L(tokenized)
    del tokenized
    gc.collect()

    if docs_done > 0:
        logger.info("Resuming BM25 sparse %s from doc %d/%d", split, docs_done, n_docs)

    def _bm25_iter(start: int):
        for i in range(start, n_docs):
            yield _sparse_vector_bm25l(bm25, i)

    _stream_sparse_to_npy(base_path, doc_ids[docs_done:], _bm25_iter(docs_done), _resume=resume)
    return n_docs


def create_sparse_embeddings_pymilvus_bm25(
    dataset,
    dataset_name: str,
    split: str,
    batch_size: int = 1000,
):
    """
    Encode with PyMilvus BM25EmbeddingFunction (client-side BM25 from pymilvus.model.sparse).
    Supports restart: resumes from last checkpoint if interrupted.

    Output: data/<dataset_name>_sparse_<split>_pymilvus_bm25_*.npy
    """
    from pymilvus.model.sparse import BM25EmbeddingFunction

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path  = out_dir / f"{dataset_name}_sparse_{split}_pymilvus_bm25"
    model_path = str(out_dir / f"{dataset_name}_pymilvus_bm25_model.json")

    if _sparse_output_complete(base_path):
        logger.info("PyMilvus BM25 sparse %s already complete — skipping", split)
        return len(dataset["text"])

    docs_done, resume = _load_sparse_checkpoint(base_path)

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    ef = BM25EmbeddingFunction()

    if split == "corpus":
        # Always fit on full corpus — IDF requires all documents
        logger.info("Fitting PyMilvus BM25 on %d corpus documents...", n_docs)
        ef.fit(texts)
        ef.save(model_path)
        logger.info("Saved PyMilvus BM25 model → %s", model_path)
        encode_fn = ef.encode_documents
    else:
        logger.info("Loading PyMilvus BM25 model from %s", model_path)
        ef.load(model_path)
        encode_fn = ef.encode_queries

    if docs_done > 0:
        logger.info("Resuming PyMilvus BM25 sparse %s from doc %d/%d", split, docs_done, n_docs)

    remaining_texts = texts[docs_done:]

    def _pymilvus_bm25_iter():
        for i in tqdm(range(0, len(remaining_texts), batch_size), desc=f"PyMilvus BM25 {split}"):
            batch = remaining_texts[i : i + batch_size]
            vecs  = encode_fn(batch)
            for j in range(len(batch)):
                row = vecs[j].tocsr()
                yield row.indices.tolist(), row.data.tolist()
            del vecs
            gc.collect()

    _stream_sparse_to_npy(base_path, doc_ids[docs_done:], _pymilvus_bm25_iter(), _resume=resume)
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
    Supports restart: resumes from last checkpoint if interrupted.

    Output: data/<dataset_name>_sparse_<split>_milvus_splade_*.npy
    """
    from pymilvus.model.sparse import SpladeEmbeddingFunction

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_milvus_splade"

    if _sparse_output_complete(base_path):
        logger.info("Milvus SPLADE sparse %s already complete — skipping", split)
        return len(dataset["text"])

    docs_done, resume = _load_sparse_checkpoint(base_path)

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    logger.info("Loading Milvus SPLADE model: %s", milvus_splade_model)
    ef        = SpladeEmbeddingFunction(model_name=milvus_splade_model, device=device)
    encode_fn = ef.encode_documents if split == "corpus" else ef.encode_queries

    if docs_done > 0:
        logger.info("Resuming Milvus SPLADE sparse %s from doc %d/%d", split, docs_done, n_docs)

    remaining_texts = texts[docs_done:]

    def _milvus_splade_iter():
        for i in tqdm(range(0, len(remaining_texts), batch_size), desc=f"Milvus SPLADE {split}"):
            batch = remaining_texts[i : i + batch_size]
            vecs  = encode_fn(batch)
            for j in range(len(batch)):
                row = vecs[j].tocsr()
                yield row.indices.tolist(), row.data.tolist()
            del vecs
            gc.collect()

    _stream_sparse_to_npy(base_path, doc_ids[docs_done:], _milvus_splade_iter(), _resume=resume)
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
    Supports restart: resumes from last checkpoint if interrupted.

    Output: data/<dataset_name>_sparse_<split>_splade_*.npy
    """
    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{dataset_name}_sparse_{split}_splade"

    if _sparse_output_complete(base_path):
        logger.info("SPLADE sparse %s already complete — skipping", split)
        return len(dataset["text"])

    docs_done, resume = _load_sparse_checkpoint(base_path)

    texts   = list(dataset["text"])
    doc_ids = list(dataset["_id"])
    n_docs  = len(texts)

    logger.info("Loading SPLADE model: %s", sparse_model_id)
    model = SparseEncoder(sparse_model_id, device=device, cache_folder=cache_dir)
    pool  = model.start_multi_process_pool(target_devices=["cpu"] * workers)

    if docs_done > 0:
        logger.info("Resuming SPLADE sparse %s from doc %d/%d", split, docs_done, n_docs)

    remaining_texts = texts[docs_done:]
    logger.info("Encoding %d texts (SPLADE) using %d workers …", len(remaining_texts), workers)

    def _splade_iter():
        for i in tqdm(range(0, len(remaining_texts), batch_size), desc=f"SPLADE {split}"):
            batch_vecs = model.encode(
                remaining_texts[i : i + batch_size], pool=pool, show_progress_bar=False
            )
            for sv in batch_vecs:
                sv = sv.coalesce()
                sentence_indices = sv.indices().tolist()
                indices = sentence_indices[1] if len(sentence_indices) == 2 else sentence_indices[0]
                yield indices, sv.values().tolist()
            del batch_vecs
            gc.collect()

    _stream_sparse_to_npy(base_path, doc_ids[docs_done:], _splade_iter(), _resume=resume)

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
