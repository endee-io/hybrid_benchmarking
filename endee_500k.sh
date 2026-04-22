python3.13 -m src.main \
  --db endee \
  --index-name quora_bench_int16 \
  --dataset-name beir_quora \
  --results run_quora_int16_topk_100 \
  --concurrency 16 \
  --top-k 100 \
  --sparse-mode endee_bm25 \
  --sparse-scoring-model endee_bm25 \
  --precision int16 \
  --vector-token localtest \
  --base-url http://148.113.58.83:8080/api/v1 \
  --validation-venv validation-env \
  --cache-dir model_cache \
  --skip-indexing

