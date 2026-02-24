import argparse
import gc
import json
import logging
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SparseEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DENSE_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_ID = "prithivida/Splade_PP_en_v1"


def load_hf_dataset(dataset_id: str, embed_type: str, cache_dir: str = None):
    """
    Load a HuggingFace BeIR-style dataset.
    - corpus  → config="corpus",  split="corpus"
    - queries → config="queries", split="queries"
    """
    config = embed_type  # "corpus" or "queries"
    logger.info("Loading dataset '%s' config='%s' split='%s'", dataset_id, config, config)
    dataset = load_dataset(dataset_id, config, split=config, cache_dir=cache_dir)
    logger.info("Loaded %d records", len(dataset))
    return dataset


def create_embeddings(
    dataset,
    output_path: str,
    batch_size: int = 1000,
    device: str = "cpu",
    dense_model_id: str = DENSE_MODEL_ID,
    sparse_model_id: str = SPARSE_MODEL_ID,
    cache_dir: str = None,
):
    """Encode every record in dataset and write JSONL to output_path."""
    logger.info("Loading dense model: %s", dense_model_id)
    dense_model  = SentenceTransformer(dense_model_id, device=device, cache_folder=cache_dir)
    logger.info("Loading sparse model: %s", sparse_model_id)
    sparse_model = SparseEncoder(sparse_model_id, device=device, cache_folder=cache_dir)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding"):
            batch = dataset[i : i + batch_size]

            texts   = batch["text"]   if isinstance(batch, dict) else [r["text"]   for r in batch]
            doc_ids = batch["_id"]    if isinstance(batch, dict) else [r["_id"]    for r in batch]

            if not isinstance(texts,   list): texts   = list(texts)
            if not isinstance(doc_ids, list): doc_ids = list(doc_ids)

            dense_vecs  = dense_model.encode(texts)
            sparse_vecs = sparse_model.encode(texts)

            for j in range(len(texts)):
                if sparse_vecs[j] is None or dense_vecs[j] is None:
                    total_skipped += 1
                    continue

                sparse_embedding = sparse_vecs[j].coalesce()
                sentence_indices = sparse_embedding.indices().tolist()
                values           = sparse_embedding.values().tolist()
                indices = sentence_indices[1] if len(sentence_indices) == 2 else sentence_indices[0]

                record = {
                    "id": doc_ids[j],
                    "dense_vector": dense_vecs[j].tolist(),
                    "sparse_vector": {"indices": indices, "values": values},
                    "meta": {"text": texts[j]},
                }
                f.write(json.dumps(record) + "\n")
                total_written += 1

            del batch, texts, doc_ids, dense_vecs, sparse_vecs
            gc.collect()

    logger.info("Done — written: %d  skipped: %d  → %s", total_written, total_skipped, output_path)


def main():
    parser = argparse.ArgumentParser(description="Create dense+sparse embeddings from a BeIR dataset.")
    parser.add_argument("--dataset-id", required=True,
                        help="HuggingFace dataset ID, e.g. 'BeIR/quora' or 'BeIR/hotpotqa'")
    parser.add_argument("--output-path", required=True,
                        help="Output JSONL file path, e.g. corpus_embeddings",default="embeddings")
    parser.add_argument("--batch-size",    type=int, default=1000,         help="Encoding batch size (default: 1000)")
    parser.add_argument("--device",        default="cpu",                  help="Torch device (default: cpu)")
    parser.add_argument("--dense-model",   default=DENSE_MODEL_ID,         help=f"Dense model ID (default: {DENSE_MODEL_ID})")
    parser.add_argument("--sparse-model",  default=SPARSE_MODEL_ID,        help=f"Sparse model ID (default: {SPARSE_MODEL_ID})")
    parser.add_argument("--cache-dir",     default=None,                   help="Local cache directory for HuggingFace datasets and models (default: HF default cache)")
    args = parser.parse_args()

    corpus_dataset = load_hf_dataset(args.dataset_id, "corpus", cache_dir=args.cache_dir)
    corpus_output_path = args.output_path+"_corpus.jsonl"
    create_embeddings(
        dataset=corpus_dataset,
        output_path=corpus_output_path,
        batch_size=args.batch_size,
        device=args.device,
        dense_model_id=args.dense_model,
        sparse_model_id=args.sparse_model,
        cache_dir=args.cache_dir,
    )
    queries_dataset = load_hf_dataset(args.dataset_id, "queries", cache_dir=args.cache_dir)
    queries_output_path = args.output_path+"_queries.jsonl"
    create_embeddings(
        dataset=queries_dataset,
        output_path=queries_output_path,
        batch_size=args.batch_size,
        device=args.device,
        dense_model_id=args.dense_model,
        sparse_model_id=args.sparse_model,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
