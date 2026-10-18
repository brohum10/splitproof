"""Synthetic duplicate-contamination demonstration; no model training."""

import json
from splitproof import split_dataset


def fixture():
    texts = [
        "volcano magma crater basalt ash lava eruption tectonic plume",
        "violin bow string concert orchestra melody rhythm harmony acoustic",
        "ocean reef coral fish tide current marine lagoon salinity",
        "database transaction index query schema rollback commit isolation cursor",
        "garden tulip soil seed petal sunlight watering compost fertilizer",
        "satellite orbit antenna telemetry launch payload spacecraft altitude navigation",
    ]
    documents = []
    for i, text in enumerate(texts):
        documents.extend([{"id": f"topic-{i}-a", "text": text, "group": None},
                          {"id": f"topic-{i}-b", "text": text.upper()+"!", "group": None}])
    return {"documents": documents, "shingle_size": 2, "threshold_percent": 70,
            "weights": {"train": 60, "validation": 20, "test": 20}, "seed": 42}


def main():
    spec = fixture()
    report = split_dataset(spec)
    naive = {d["id"]: "train" if i % 2 == 0 else "test" for i, d in enumerate(spec["documents"])}
    before = sum(naive[p["a"]] != naive[p["b"]] for p in report["similarity_links"])
    assert before == 6
    assert report["cross_split_link_count"] == 0
    assert all(report["counts"].values())
    print(json.dumps({"synthetic_demo": True, "documents": 12, "duplicate_families": 6,
                      "naive_cross_split_links": before,
                      "splitproof_cross_split_links": report["cross_split_link_count"],
                      "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
