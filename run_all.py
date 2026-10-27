"""CLI for the same coding-only experiment service used by the local UI."""

import argparse

from src.coding_problems import get_problems
from src.model_loader import DEFAULT_MODEL_NAME
from src.web_service import ExperimentService, RunRequest


def main():
    parser = argparse.ArgumentParser(description="Compare original/pruned Python test outcomes (no judge).")
    parser.add_argument("--category", default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, choices=[DEFAULT_MODEL_NAME, "gpt2"])
    parser.add_argument("--problems", type=int, default=10)
    parser.add_argument("--allow-execution", action="store_true", help="Consent to restricted local Python child processes; not a full security sandbox.")
    args = parser.parse_args()
    if not args.allow_execution:
        parser.error("Read README.md, then pass --allow-execution to consent to local Python execution.")
    problems = get_problems(args.category)[:args.problems]
    if not 1 <= args.problems <= 20 or not problems:
        parser.error("Select 1–20 problems from a valid category.")
    request = RunRequest(topic=args.category, model_name=args.model, problems=problems, execution_consent=True,
                         variants=[{"technique":"layers", "strength":1},
                                   {"technique":"heads", "strength":3},
                                   {"technique":"magnitude", "strength":20}])
    service = ExperimentService()
    run_id = service.start(request)
    last = None
    try:
        while service.worker.is_alive():
            record = service.snapshot(run_id)
            if record["stage"] != last:
                print(record["stage"], flush=True)
                last = record["stage"]
            service.worker.join(timeout=1)
    except KeyboardInterrupt:
        service.cancel(run_id)
        service.worker.join()
    record = service.snapshot(run_id)
    print(f"\nStatus: {record['status']}")
    if record["error"]:
        print(record["error"])
    for vi, variant in enumerate(request.variants):
        cells = [value for key,value in record["comparisons"].items() if key.startswith(f"{vi}:")]
        print(f"{variant.label}: {sum(c['score'] for c in cells)}/{len(cells)} matching tested outcomes")
    print(f"Saved: results/web_runs/{run_id}.json")


if __name__ == "__main__":
    main()
