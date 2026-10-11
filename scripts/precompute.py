#!/usr/bin/env python3
"""Precompute every result the BeeFAME website can show, so it is served from cache.

Walks every combination the UI offers (datasets x classifiers x mitigation methods
x target subgroups, plus every Beespector Deep Dive configuration) and requests each
one from the running services. Results land in Redis and in Beespector's model store.

Safe to re-run: finished jobs are cache hits and return immediately. Jobs run
cheapest first, so the slow Adult + SVC + in-processing runs come last.

Run it next to the services, not through nginx (which times out after 60s):

    python3 scripts/precompute.py                    # everything
    python3 scripts/precompute.py --dry-run          # list the jobs only
    python3 scripts/precompute.py --dataset german   # one dataset
    python3 scripts/precompute.py --max-tier 3       # skip the slowest jobs
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

TEST_SIZE = 0.2  # the website locks test size at 0.2
INPROCESSING = {"Exponentiated Gradient", "Grid Search"}
SVC = "Support Vector Classification (SVC)"
STAGE_ORDER = {"analysis": 0, "evaluation": 1, "beespector": 2, "target": 3}


def http(method, url, payload=None, timeout=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def browser_params(classifier):
    """The params the browser sends: every default, integral floats as ints (JSON.stringify(1.0) is "1")."""
    params = {}
    for p in classifier["params"]:
        value = p.get("default")
        if value is None:
            value = ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        params[p["title"]] = value
    return params


def beespector_classifier_slug(name):
    # Mirrors frontend/src/pages/beespector.tsx
    return re.sub(r"\s+", "_", name.lower()).replace("_(svc)", "", 1).replace("_classifier", "", 1)


def beespector_sensitive_feature(dataset):
    # The Deep Dive page always uses the dataset's first sensitive feature
    name = dataset["sensitive_features"][0]["name"].lower()
    return "sex" if name == "gender" else name


def tier(dataset, classifier, method):
    """0 is cheapest. Adult is ~50x German; SVC and in-processing are each far slower again."""
    heavy = (method in INPROCESSING) + (classifier == SVC)
    return heavy * 2 + (dataset == "adult")


def discover_targets(backend, dataset, classifiers, pairs, timeout):
    """Every target subgroup the UI can offer, read from one cheap analysis run."""
    cheapest = next(c for c in classifiers if c["name"] == "Logistic Regression")
    response = http("POST", f"{backend}/analysis/", {
        "dataset_names": [dataset],
        "classifiers": [{"name": cheapest["name"], "params": browser_params(cheapest)}],
        "test_size": TEST_SIZE,
        "subgroup_pairs": pairs,
    }, timeout=timeout)
    by_label = {p["label"]: p for p in pairs}
    targets = []
    for row in response["data"]:
        pair = by_label.get(row.get("Sensitive Column"))
        if not row.get("Is Subgroup") or pair is None:
            continue
        for group_label in (row.get("Per Group Metrics") or {}):
            targets.append({"col1": pair["col1"], "col2": pair["col2"],
                            "col3": pair.get("col3"), "group_label": group_label})
    return targets


def build_jobs(args):
    backend, beespector = args.backend.rstrip("/"), args.beespector.rstrip("/")
    datasets = http("GET", f"{backend}/datasets/", timeout=60)["data"]
    classifiers = http("GET", f"{backend}/classifiers/", timeout=60)["data"]
    methods = [m["name"] for m in http("GET", f"{backend}/methods/", timeout=60)["data"]]

    jobs = []
    for ds in datasets:
        slug = ds["slug"]
        if args.dataset and slug not in args.dataset:
            continue
        pairs = http("GET", f"{backend}/datasets/{slug}/subgroup-pairs", timeout=600)["data"]["pairs"]
        pairs = [{"col1": p["col1"], "col2": p["col2"], "col3": p.get("col3"), "label": p["label"]} for p in pairs]
        targets = discover_targets(backend, slug, classifiers, pairs, args.timeout) if pairs else []
        print(f"{slug}: {len(pairs)} subgroup pairs, {len(targets)} target subgroups", flush=True)

        for clf in classifiers:
            name = clf["name"]
            jobs.append(dict(stage="analysis", dataset=slug, classifier=name, method=None, target=None,
                             url=f"{backend}/analysis/", payload={
                                 "dataset_names": [slug],
                                 "classifiers": [{"name": name, "params": browser_params(clf)}],
                                 "test_size": TEST_SIZE, "subgroup_pairs": None}))
            for method in methods:
                for target in [None] + targets:
                    jobs.append(dict(stage="target" if target else "evaluation", dataset=slug,
                                     classifier=name, method=method, target=target,
                                     url=f"{backend}/evaluation/", payload={
                                         "dataset_names": [slug], "classifier_names": [name],
                                         "method_names": [method], "test_size": TEST_SIZE,
                                         "subgroup_pairs": None, "target_subgroup": target}))
            for method in ["None"] + methods:
                jobs.append(dict(stage="beespector", dataset=slug, classifier=name, method=method, target=None,
                                 url=f"{beespector}/api/initialize_context", payload={
                                     "dataset_name": slug,
                                     "base_classifier": beespector_classifier_slug(name),
                                     "classifier_params": browser_params(clf),
                                     "mitigation_method": re.sub(r"\s+", "_", method.lower()),
                                     "sensitive_feature": beespector_sensitive_feature(ds)}))

    for job in jobs:
        job["tier"] = tier(job["dataset"], job["classifier"], job["method"])
    jobs = [j for j in jobs if j["tier"] <= args.max_tier and (not args.stage or j["stage"] in args.stage)]
    jobs.sort(key=lambda j: (j["tier"], STAGE_ORDER[j["stage"]]))
    return jobs


def describe(job):
    target = f" -> {job['target']['group_label']}" if job["target"] else ""
    method = f" / {job['method']}" if job["method"] else ""
    return f"[t{job['tier']}] {job['stage']:10} {job['dataset']:6} {job['classifier']}{method}{target}"


def run_job(job, timeout):
    started = time.time()
    try:
        http("POST", job["url"], job["payload"], timeout=timeout)
        return "ok", time.time() - started, None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        return "fail", time.time() - started, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return "fail", time.time() - started, f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--beespector", default="http://127.0.0.1:8001")
    parser.add_argument("--dataset", action="append", help="only this dataset slug (repeatable)")
    parser.add_argument("--stage", action="append", choices=list(STAGE_ORDER), help="only this stage (repeatable)")
    parser.add_argument("--max-tier", type=int, default=5, help="skip jobs above this cost tier (0-5)")
    parser.add_argument("--workers", type=int, default=2, help="parallel requests (Beespector serialises its own)")
    parser.add_argument("--timeout", type=float, default=48 * 3600,
                        help="seconds per request; keep it long, a client timeout does not stop the server")
    parser.add_argument("--log", default="precompute.jsonl", help="one JSON line per finished job")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args)
    print(f"{len(jobs)} jobs:", dict(Counter((f"t{j['tier']}", j["stage"]) for j in jobs)), flush=True)
    if args.dry_run:
        for job in jobs:
            print(describe(job))
        return 0

    started, failures = time.time(), 0
    with open(args.log, "a") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, args.timeout): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            status, seconds, error = future.result()
            failures += status != "ok"
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp} {done}/{len(jobs)} {status.upper():4} {seconds:8.1f}s {describe(job)}"
                  + (f"\n    {error}" if error else ""), flush=True)
            log.write(json.dumps({"time": stamp, "status": status, "seconds": round(seconds, 1), "error": error,
                                  **{k: job[k] for k in ("tier", "stage", "dataset", "classifier", "method", "target")}})
                      + "\n")
            log.flush()

    hours = (time.time() - started) / 3600
    print(f"Finished {len(jobs)} jobs in {hours:.1f}h, {failures} failed.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
