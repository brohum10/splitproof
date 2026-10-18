"""Leakage-aware document splitting with exact token-shingle Jaccard links."""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata

MAX_BYTES = 3_000_000
MAX_CANDIDATES = 100_000
MAX_COMPARE_WORK = 10_000_000
SPLITS = ("train", "validation", "test")


class InvalidDataset(ValueError):
    pass


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise InvalidDataset(f"{name} must be an integer in [{low}, {high}]")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value):
        raise InvalidDataset("IDs must use 1–100 ASCII letters, digits, dots, underscores or hyphens")
    return value


def fields(obj, expected):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise InvalidDataset("incorrect fields; expected " + ", ".join(expected))


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidDataset(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    with open(path, "rb") as handle:
        raw = handle.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES:
        raise InvalidDataset("input exceeds 3 MB")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InvalidDataset("invalid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise InvalidDataset("input must be a JSON object")
    return result


def shingles(text, size):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    if not words:
        raise InvalidDataset("document has no word tokens")
    if len(words) > 10_000:
        raise InvalidDataset("document exceeds 10,000 word tokens")
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[i:i+size]) for i in range(len(words)-size+1)}


def validate(spec):
    fields(spec, ("documents", "shingle_size", "threshold_percent", "weights", "seed"))
    integer(spec["shingle_size"], "shingle_size", 1, 5)
    integer(spec["threshold_percent"], "threshold_percent", 1, 100)
    integer(spec["seed"], "seed", 0, 2**32-1)
    fields(spec["weights"], SPLITS)
    for split in SPLITS:
        integer(spec["weights"][split], split+" weight", 0, 100)
    if spec["weights"]["train"] == 0 or spec["weights"]["test"] == 0:
        raise InvalidDataset("train and test weights must be positive")
    documents = spec["documents"]
    if not isinstance(documents, list) or not 1 <= len(documents) <= 1000:
        raise InvalidDataset("documents must contain 1–1000 entries")
    ids, features = set(), {}
    for doc in documents:
        fields(doc, ("id", "text", "group"))
        identifier(doc["id"])
        if doc["id"] in ids:
            raise InvalidDataset("duplicate document ID")
        ids.add(doc["id"])
        if doc["group"] is not None:
            identifier(doc["group"])
        if not isinstance(doc["text"], str) or len(doc["text"].encode("utf-8")) > 50_000:
            raise InvalidDataset("text must be a string of at most 50 KB")
        features[doc["id"]] = shingles(doc["text"], spec["shingle_size"])
    if sum(map(len, features.values())) > 100_000:
        raise InvalidDataset("dataset exceeds 100,000 distinct per-document shingles")
    return sorted(documents, key=lambda d: d["id"]), features


def similar_pairs(features, threshold):
    """Enumerate all pairs meeting positive Jaccard threshold; no LSH misses."""
    postings = defaultdict(list)
    for key in sorted(features):
        for feature in sorted(features[key]):
            postings[feature].append(key)
    if sum(len(ids)*(len(ids)-1)//2 for ids in postings.values()) > 2_000_000:
        raise InvalidDataset("posting expansion work limit exceeded")
    candidates = set()
    for members in postings.values():
        for pair in itertools.combinations(members, 2):
            candidates.add(pair)
            if len(candidates) > MAX_CANDIDATES:
                raise InvalidDataset("candidate pair limit exceeded; no partial split produced")
    # Bound intersections before doing them. A rejected run never silently samples.
    work = sum(min(len(features[a]), len(features[b])) for a, b in candidates)
    if work > MAX_COMPARE_WORK:
        raise InvalidDataset("exact comparison work limit exceeded")
    matches = []
    for a, b in sorted(candidates):
        left, right = features[a], features[b]
        # Length bound: even a complete smaller-set overlap could not qualify.
        if min(len(left), len(right))*100 < max(len(left), len(right))*threshold:
            continue
        intersection = len(left & right)
        union = len(left)+len(right)-intersection
        if intersection*100 >= threshold*union:
            matches.append({"a": a, "b": b, "intersection": intersection, "union": union})
    return matches, len(candidates)


class Components:
    def __init__(self, ids):
        self.parent = {key: key for key in ids}

    def find(self, key):
        while key != self.parent[key]:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def seeded_key(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts], separators=(",", ":")).encode()).hexdigest()


def split_dataset(spec):
    documents, features = validate(spec)
    pairs, candidate_count = similar_pairs(features, spec["threshold_percent"])
    components = Components(features)
    for pair in pairs:
        components.join(pair["a"], pair["b"])
    first_in_group, group_links = {}, []
    for doc in documents:
        group = doc["group"]
        if group is not None:
            if group in first_in_group:
                other = first_in_group[group]
                components.join(other, doc["id"])
                group_links.append({"a": other, "b": doc["id"], "group": group})
            else:
                first_in_group[group] = doc["id"]
    members = defaultdict(list)
    for doc in documents:
        members[components.find(doc["id"])].append(doc["id"])
    ordered = sorted(members.values(), key=lambda ids: (-len(ids), seeded_key(spec["seed"], ids)))
    total_weight = sum(spec["weights"].values())
    n = len(documents)
    counts = dict.fromkeys(SPLITS, 0)
    assignment, component_rows = {}, []
    for ids in ordered:
        size = len(ids)
        # Minimize the change in squared target-count error. Integer arithmetic
        # makes ties reproducible; this placement heuristic is not ratio-optimal.
        def placement(split):
            delta = ((counts[split]+size)**2-counts[split]**2)*total_weight
            delta -= 2*n*spec["weights"][split]*size
            return delta, seeded_key(spec["seed"], ids, split)
        destination = min((s for s in SPLITS if spec["weights"][s] > 0), key=placement)
        counts[destination] += size
        component_id = min(ids)
        for key in ids:
            assignment[key] = (destination, component_id)
        component_rows.append({"id": component_id, "members": ids, "split": destination})
    # Explicitly verify the graph constraints before exporting the manifest.
    violations = [p for p in pairs+group_links if assignment[p["a"]][0] != assignment[p["b"]][0]]
    if violations:
        raise RuntimeError("internal split verification failed")
    normalized = {**spec, "documents": documents}
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    warnings = [f"{s} is empty; review component sizes and requested weights"
                for s in SPLITS if spec["weights"][s] > 0 and counts[s] == 0]
    return {"version": 1, "input_sha256": digest,
            "config": {k: spec[k] for k in ("shingle_size", "threshold_percent", "weights", "seed")},
            "counts": counts, "targets": {s: n*spec["weights"][s]/total_weight for s in SPLITS},
            "cross_split_link_count": len(violations), "candidate_pair_count": candidate_count,
            "similarity_links": pairs, "explicit_group_links": group_links,
            "components": sorted(component_rows, key=lambda c: c["id"]), "warnings": warnings,
            "documents": [{"id": d["id"], "split": assignment[d["id"]][0],
                           "component": assignment[d["id"]][1],
                           "text_sha256": hashlib.sha256(d["text"].encode()).hexdigest()} for d in documents]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        report = split_dataset(load(args.input))
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc), "kind": "invalid_dataset"}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
