import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from demo import fixture
from splitproof import InvalidDataset, load, shingles, similar_pairs, split_dataset


def document(key, text, group=None):
    return {"id": key, "text": text, "group": group}


def spec(documents):
    return {"documents": documents, "shingle_size": 1, "threshold_percent": 60,
            "weights": {"train": 60, "validation": 20, "test": 20}, "seed": 42}


def exhaustive(features, threshold):
    result = []
    for a, b in itertools.combinations(sorted(features), 2):
        left, right = features[a], features[b]
        overlap = sum(feature in right for feature in left)
        union = len(left)+len(right)-overlap
        if overlap*100 >= threshold*union:
            result.append({"a": a, "b": b, "intersection": overlap, "union": union})
    return result


class SplitTests(unittest.TestCase):
    def test_synthetic_demo(self):
        report = split_dataset(fixture())
        self.assertEqual(len(report["similarity_links"]), 6)
        self.assertEqual(len(report["components"]), 6)
        self.assertEqual(report["cross_split_link_count"], 0)
        self.assertTrue(all(report["counts"].values()))

    def test_index_matches_exhaustive_randomized(self):
        rng = random.Random(18)
        for case in range(300):
            features = {str(i): {(str(j),) for j in rng.sample(range(15), rng.randint(1, 8))}
                        for i in range(rng.randint(1, 15))}
            threshold = rng.randint(1, 100)
            with self.subTest(case=case):
                actual, _ = similar_pairs(features, threshold)
                self.assertEqual(actual, exhaustive(features, threshold))

    def test_transitive_chain(self):
        report = split_dataset(spec([document("a", "one two"), document("b", "one two three"),
                                     document("c", "two three")]))
        self.assertEqual(len(report["similarity_links"]), 2)
        self.assertEqual(len(report["components"]), 1)
        self.assertEqual(report["components"][0]["members"], ["a", "b", "c"])

    def test_explicit_group_overrides_text_difference(self):
        report = split_dataset(spec([document("a", "red green", "thread"),
                                     document("b", "blue yellow", "thread")]))
        self.assertEqual(len(report["similarity_links"]), 0)
        self.assertEqual(len(report["components"]), 1)
        self.assertEqual(report["explicit_group_links"][0]["group"], "thread")

    def test_group_and_similarity_chain(self):
        report = split_dataset(spec([document("a", "red green", "thread"),
                                     document("b", "blue yellow", "thread"),
                                     document("c", "blue yellow")]))
        self.assertEqual(len(report["components"]), 1)

    def test_normalization(self):
        self.assertEqual(shingles("CAFÉ, ＨＥＬＬＯ!", 2), shingles("café hello", 2))

    def test_short_document(self):
        self.assertEqual(shingles("one two", 3), {("one", "two")})
        self.assertEqual(shingles("one", 3), {("one",)})

    def test_threshold_boundary_inclusive(self):
        features = {"a": {("x",), ("y",)}, "b": {("y",), ("z",)}}
        self.assertEqual(len(similar_pairs(features, 33)[0]), 1)
        self.assertEqual(len(similar_pairs(features, 34)[0]), 0)

    def test_exact_threshold(self):
        features = {"a": {("x",)}, "b": {("x",)}, "c": {("x",), ("y",)}}
        self.assertEqual(similar_pairs(features, 100)[0], [{"a": "a", "b": "b", "intersection": 1, "union": 1}])

    def test_disjoint_documents_generate_no_candidates(self):
        report = split_dataset(spec([document("a", "a b"), document("b", "c d")]))
        self.assertEqual(report["candidate_pair_count"], 0)

    def test_every_independent_oracle_link_stays_inside_split(self):
        rng = random.Random(42)
        docs = [document(str(i), " ".join(str(x) for x in rng.sample(range(8), 4))) for i in range(40)]
        report = split_dataset(spec(docs))
        assignment = {d["id"]: d["split"] for d in report["documents"]}
        features = {d["id"]: shingles(d["text"], 1) for d in docs}
        for edge in exhaustive(features, 60):
            self.assertEqual(assignment[edge["a"]], assignment[edge["b"]])

    def test_reordered_input_identical(self):
        data = fixture()
        expected = split_dataset(data)
        data["documents"].reverse()
        self.assertEqual(expected, split_dataset(data))

    def test_does_not_mutate_input(self):
        data = fixture()
        original = copy.deepcopy(data)
        split_dataset(data)
        self.assertEqual(data, original)

    def test_same_seed_reproducible(self):
        self.assertEqual(split_dataset(fixture()), split_dataset(fixture()))

    def test_input_fingerprint_changes(self):
        data = fixture()
        first = split_dataset(data)
        data["seed"] = 8
        self.assertNotEqual(first["input_sha256"], split_dataset(data)["input_sha256"])

    def test_text_hash_exact(self):
        text = "A word.\nAnother word."
        report = split_dataset(spec([document("a", text)]))
        self.assertEqual(report["documents"][0]["text_sha256"], hashlib.sha256(text.encode()).hexdigest())

    def test_single_component_warns_empty_splits(self):
        report = split_dataset(spec([document("a", "one"), document("b", "one")]))
        self.assertEqual(len(report["warnings"]), 2)
        self.assertEqual(sum(report["counts"].values()), 2)

    def test_zero_validation_weight(self):
        data = fixture()
        data["weights"]["validation"] = 0
        report = split_dataset(data)
        self.assertEqual(report["counts"]["validation"], 0)
        self.assertFalse(any(w.startswith("validation") for w in report["warnings"]))

    def test_invalid_config_numbers(self):
        for field in ("shingle_size", "threshold_percent", "seed"):
            for value in (True, -1, "2", None, 1.5):
                data = fixture()
                data[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(InvalidDataset): split_dataset(data)

    def test_zero_threshold_rejected(self):
        data = fixture()
        data["threshold_percent"] = 0
        with self.assertRaises(InvalidDataset): split_dataset(data)

    def test_train_and_test_required(self):
        for split in ("train", "test"):
            data = fixture()
            data["weights"][split] = 0
            with self.assertRaises(InvalidDataset): split_dataset(data)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(InvalidDataset): split_dataset(spec([document("a", "one"), document("a", "two")]))

    def test_invalid_ids_and_groups(self):
        for invalid in ("", "a b", "<script>", "x"*101, True):
            with self.assertRaises(InvalidDataset): split_dataset(spec([document(invalid, "one")]))
            with self.assertRaises(InvalidDataset): split_dataset(spec([document("a", "one", invalid)]))

    def test_blank_or_nonword_text_rejected(self):
        for value in ("", "   ", "!!!", None, 3):
            with self.assertRaises(InvalidDataset): split_dataset(spec([document("a", value)]))

    def test_text_and_word_limits(self):
        for value in ("a"*50001, "x "*10001):
            with self.assertRaises(InvalidDataset): split_dataset(spec([document("a", value)]))

    def test_unknown_fields_rejected(self):
        data = fixture()
        data["documents"][0]["split"] = "test"
        with self.assertRaises(InvalidDataset): split_dataset(data)

    def test_empty_and_large_datasets_rejected(self):
        with self.assertRaises(InvalidDataset): split_dataset(spec([]))
        with self.assertRaises(InvalidDataset): split_dataset(spec([document(str(i), "word") for i in range(1001)]))

    def test_candidate_limit_fails_without_partial_result(self):
        with patch("splitproof.MAX_CANDIDATES", 1):
            with self.assertRaisesRegex(InvalidDataset, "candidate pair"):
                similar_pairs({str(i): {("x",)} for i in range(3)}, 70)

    def test_comparison_work_limit(self):
        with patch("splitproof.MAX_COMPARE_WORK", 0):
            with self.assertRaisesRegex(InvalidDataset, "comparison work"):
                similar_pairs({"a": {("x",)}, "b": {("x",)}}, 70)

    def test_posting_expansion_limit(self):
        features = {str(i): {(str(j),) for j in range(500)} for i in range(100)}
        with self.assertRaisesRegex(InvalidDataset, "posting expansion"):
            similar_pairs(features, 70)

    def test_invalid_encoding_and_nested_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.json"
            for content in (b'\xff', b'['*2000+b']'*2000):
                path.write_bytes(content)
                with self.assertRaises(InvalidDataset): load(path)

    def test_duplicate_json_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.json"
            path.write_text('{"a":1,"a":2}')
            with self.assertRaises(InvalidDataset): load(path)
            with patch("splitproof.MAX_BYTES", 2):
                with self.assertRaises(InvalidDataset): load(path)

    def test_cli_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.json"
            for data, code in ((fixture(), 0), ({}, 2)):
                path.write_text(json.dumps(data))
                result = subprocess.run([sys.executable, "splitproof.py", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, code)
                self.assertIsInstance(json.loads(result.stdout if code == 0 else result.stderr), dict)


if __name__ == "__main__":
    unittest.main()
