import sys

# Peek at --vector-db before Click takes over argument parsing
vector_db = "endee"
for i, arg in enumerate(sys.argv[1:], 1):
    if arg.startswith("--vector-db="):
        vector_db = arg.split("=", 1)[1]
        sys.argv.pop(i)
        break
    if arg == "--vector-db" and i < len(sys.argv) - 1:
        vector_db = sys.argv[i + 1]
        sys.argv.pop(i + 1)
        sys.argv.pop(i)
        break

if vector_db == "qdrant":
    from src.qdrant_benchmark import sparse_vector_benchmark
else:
    from src.benchmark import sparse_vector_benchmark

# main function
if __name__ == "__main__":
    sparse_vector_benchmark()
