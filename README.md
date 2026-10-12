# Structure-aware semantic source-code similarity search

CSE 881 final project, Fall 2026  
Milad Rabizadeh, Vanshika Kadian, Xunlei Qian.

Source-code similarity search retrieves programs that solve the same task even when their syntax and implementation differ. We build a similarity method that combines normalized lexical features with structure from abstract syntax trees, using Project CodeNet (about 14 million submissions across 4,000+ problems). Two accepted programs count as similar when they solve the same problem, which is a practical stand-in for functional similarity.

We evaluate the method as a retrieval system against token-based similarity and pretrained CodeBERT embeddings, using Precision@k, Recall@k, MRR and MAP. We also test whether structural information helps when identifiers are renamed. A small web demo lets you paste code and see the closest programs with their scores.

## Data

The cleaned dataset is already in `data/processed/dataset.parquet` (id, problem_id, user_id, code, split), with counts after each filter in `data_card.md`. You only need the steps below to rebuild it.

Download Project CodeNet (7.8 GB, metadata included) into `data/raw/`:

    https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz

Extract only the Python files and metadata, then build from the repo root:

    cd data/raw
    tar -xzf Project_CodeNet.tar.gz 'Project_CodeNet/data/*/Python/*' 'Project_CodeNet/metadata/*'
    cd ../..
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python src/data/build_dataset.py --root data/raw/Project_CodeNet

The run is seeded, so the same raw data gives the same file.

`split` is `test`, `val` or `corpus`. Test and val are held-out queries, corpus is what gets searched. A match means the same `problem_id`. Queries come only from users with a single program in that problem, so a query never has another program by the same user in the corpus.
