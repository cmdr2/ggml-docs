# Regenerating the docs

**Pre-requisite:** `pip install libclang`

Run these commands in the root of the `ggml-docs` repo.

1. Convert the source to an intermediate JSON:
```bash
python scripts/source_to_ir.py --source-dir /path/to/ggml/include --output-dir intermediate
```
2. Update `intermediate/descriptions.json` or `intermediate/categories.json` manually if necessary.
3. Convert the intermediate JSON to markdown (remember to update the commit id in the `--link_prefix` argument!):
```bash
python scripts/ir_to_doc.py --source-dir intermediate --output-dir . --link-prefix "https://github.com/ggml-org/ggml/blob/9a4acb3/include/"
```
