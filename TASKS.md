## Oct 12 - Oct 18

- Dataset: filter CodeNet to accepted Python, clean, split (done)
- Agree on Python only, subset size, split, and the shared search function
- Metrics code: Precision@k, Recall@k, MRR, MAP, R-precision, with a small test
- Token features, first version: tokenize, mask identifiers and literals, n-grams, TF-IDF
- TF-IDF baseline on raw tokens

## Oct 18 - Oct 25

- CodeBERT embeddings (mean pooling and CLS), FAISS index, search function
- AST features, first version: node-type n-grams, subtree hashes or path contexts
- Run the baselines through the metrics code on validation queries

## Oct 25 - Oct 29

- Run baselines on test queries and make the first results table
- Combine token and AST scores, first pass at tuning the weight on validation queries

## Oct 29 - Nov 1

- Intermediate report (2 to 3 pages): problem, data, methods so far, first results. Due Nov 1

## November

- Tune the combined score, ablations: token only, AST only, combined
- Robustness tests: identifier renaming, whitespace and comment changes, dead code. Report the MAP drop
- Significance tests and error analysis
- Related work reading
- Demo app: FastAPI search endpoint and a page to paste code and see matches. CodeBERT first, then the combined score
- Spot check 20 parsed programs each

## Early December

- Final results tables and figures
- Demo polish
- Presentation, Dec 8 or 10
- Final report: ACM format, 5+ pages, 5+ references, related work, code on GitHub. Due Dec 13
- Repo cleanup and README update
