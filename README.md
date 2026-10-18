# SplitProof

**Keep related documents together before measuring model performance.**

[![Tests](https://github.com/brohum10/splitproof/actions/workflows/tests.yml/badge.svg)](https://github.com/brohum10/splitproof/actions/workflows/tests.yml)

An offline dataset splitter that links documents by token-shingle Jaccard similarity and explicit source groups, then assigns whole connected components to train, validation, or test. It exports a reproducible manifest explaining every link and assignment.

Python 3.12+ · standard library only · no API keys · no network calls

## Run it

Download this repository, open its folder, and run:

```sh
python demo.py
python splitproof.py example.json > report.json
python -m unittest -v
```

The synthetic demo has 12 documents in six duplicate families. A deliberately naive alternating train/test assignment separates each pair. SplitProof assigns each family together:

| Check | Synthetic result |
| --- | ---: |
| Documents | 12 |
| Duplicate families | 6 |
| Similarity links crossing naive train/test split | 6 |
| Similarity links crossing SplitProof partitions | 0 |
| Final train / validation / test counts | 8 / 2 / 2 |

These are correctness observations on a constructed fixture. No model was trained; no accuracy improvement, production leakage rate, or throughput has been measured.

## Why components matter

If A is similar to B and B is similar to C, separating A and C can still expose information through B. SplitProof merges the full chain, even if A and C do not directly reach the threshold. Conversation IDs, document families, or other explicit group IDs can connect otherwise dissimilar texts.

```text
documents → normalized word shingles → inverted index → exact Jaccard links
                                                               │
explicit source groups ──────────────────────────────────────────┤
                                                               ▼
                                                   connected components
                                                               │
                                                   seeded group placement
                                                               │
                                             verified partition manifest
```

The invariant is precise: every detected Jaccard link and every explicit group link stays within one partition. For completed runs, the inverted index enumerates all pairs that meet the configured positive threshold under this representation. Runs that exceed resource limits fail instead of silently omitting comparisons.

## Input

[example.json](example.json) is a complete synthetic dataset. Each document requires exactly:

```json
{"id":"doc-001","text":"The document content goes here.","group":"conversation-001"}
```

Use `group: null` when no explicit grouping is needed. Document IDs must be unique. Group IDs intentionally repeat when related records must stay together. IDs contain 1–100 ASCII letters, digits, dots, underscores, or hyphens.

The root object contains exactly `documents`, `shingle_size`, `threshold_percent`, `weights`, and `seed`. Train and test weights must be positive; validation may be zero. Weights are relative integers, not a requirement to sum to 100. The seed controls reproducible component order and placement ties. Reordering input records leaves the report unchanged.

### Similarity definition

1. Normalize text with Unicode NFKC, then casefold.
2. Extract Python Unicode `\w+` tokens, including numbers and underscores.
3. Form a **set** of consecutive token tuples of size 1–5. Repeated shingles count once. For texts shorter than that size, the entire token sequence is one feature.
4. Compare Jaccard similarity: `intersection / union`.
5. Link when `100 × intersection >= threshold_percent × union`.

Thresholds range from 1 through 100. Integer comparisons avoid floating-point ambiguity at the cutoff. At 100, equal **shingle sets** link; equal sets do not necessarily mean identical word order or identical original text. Case and punctuation variants can normalize to the same representation. Empty or punctuation-only documents are rejected.

An inverted index proposes pairs sharing at least one shingle. Since the threshold is positive, a qualifying pair must share a shingle. Every candidate is checked exactly, with a safe length bound to skip impossible matches. This is deterministic indexing, not approximate LSH.

## Placement and output

Union-find builds components from similarity and explicit group links. Components are placed largest first. A seeded hash breaks equal-size ordering ties. Each placement minimizes the incremental squared error from target document counts; integer arithmetic and seeded split ties make it reproducible.

**Placement is a heuristic, not an optimal ratio solver.** Related documents stay together even when doing so misses target ratios. Large connected components may produce an empty test or validation split; the report emits warnings. Review those warnings before evaluating a model. There is no label stratification or temporal ordering.

The JSON report contains:

- Final counts, fractional target counts, and empty-partition warnings.
- Similarity links with exact intersection/union counts.
- Explicit group links and complete component membership.
- One assignment per document with its original-text SHA-256 hash.
- A hash of normalized input ordering and all configuration.
- A checked count of links crossing partitions.

Original document text is omitted from the report. IDs, group labels, links, and hashes can still be sensitive. Use local private input paths and avoid publishing real datasets. Hashes detect changes; they are not authentication or anonymization.

Use the manifest to join source IDs back to your own data loader. Apply the split before fitting token vocabularies, preprocessing, retrieval indexes, or models. This tool exports assignments; it does not train models or copy source files into folders.

Successful CLI runs exit `0`; validation or resource-limit failures exit `2`, with JSON errors on stderr. Parser usage errors also exit `2`. Warnings remain successful runs, so downstream pipelines should explicitly inspect them.

## Tests and resource bounds

**33 tests** run locally and in GitHub Actions on Python 3.12 and 3.14. An independent all-pairs implementation checks indexed similarity results across **300 seeded randomized cases**. Additional tests verify no oracle-discovered link crosses partitions, transitive chains, explicit groups, inclusive thresholds, normalization, input ordering, text fingerprints, invalid data, resource limits, and real CLI exit codes.

Bounds: 3 MB input files, 1,000 documents, 50 KB UTF-8 text/document, 10,000 word tokens/document, and 100,000 distinct per-document shingles in total. Candidate generation is limited to 2 million posting-pair expansions and 100,000 unique candidate pairs. Exact-intersection work is bounded by 10 million summed smaller-set sizes. These are algorithmic limits, not hard wall-clock or memory deadlines. Direct library callers must bound total input object size themselves.

For sparse overlap the index avoids checking every document pair. Boilerplate-heavy collections can still be expensive and are deliberately rejected when bounds are exceeded. No production scaling benchmark is claimed.

## Limitations

- This checks lexical overlap under one representation, not semantic equivalence, paraphrase detection, factual leakage, or contamination in a pretrained model.
- Shared boilerplate may create false links and large transitive components. Review the evidence and choose a threshold appropriate to the corpus before evaluating results.
- Missing source IDs, user identities, or conversation grouping can leave nonlexical leakage undetected. Supply explicit groups when known.
- Non-whitespace languages, code, and highly templated text may require a different tokenizer/representation. Current tokenization is not model-specific.
- There is no automatic threshold tuning, class balancing, time-aware split, embedding model, live model integration, or hosted service.

The name describes the inspectable split evidence. It is not a claim that every possible form of dataset leakage has been eliminated.

## License

MIT. See [LICENSE](LICENSE).
