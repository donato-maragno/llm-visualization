"""Start the local-only LLM visualization UI using the existing Python environment."""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM visualization — open http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # No reload/multiple workers: GPU state and pruning masks must be serialized.
    uvicorn.run("web_app:app", host="127.0.0.1", port=args.port, workers=1)
