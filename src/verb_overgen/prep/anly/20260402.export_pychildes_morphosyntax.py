import polars as pl
import pyarrow as pa

dataset_original = pl.read_parquet("./data/pychildes/pychildes-morphosyntax-tokenized.20251024.parquet")

dataset = dataset_original.select(
    pl.col("id_file"),
    pl.col("id_utterance"),
    pl.col("speaker"),
    pl.col("morph").struct.field("lemma", "metadata").name.prefix("morph_"),
    pl.col("morph").struct.field("pos").cast(pl.List(pl.Categorical)).name.prefix("morph_"),
    pl.col("morph").struct.field("kind").alias("word_type").cast(pl.List(pl.Enum(["word", "compound", "punctuation"]))).name.prefix("morph_"),
    pl.col("syntax").struct.field("to_index").alias("arc").cast(pl.List(pl.UInt64)).name.prefix("syntax_"),
    pl.col("syntax").struct.field("rel").cast(pl.List(pl.Categorical)).name.prefix("syntax_"),
    pl.col("content_raw").alias("utterance"),
)

# Fields with variable-length content — keep as large_string/large_list
_KEEP_LARGE = {"morph_metadata", "utterance"}

def _optimize_field(f: pa.Field) -> pa.Field:
    t = f.type
    if t == pa.large_utf8():
        return pa.field(f.name, pa.utf8(), nullable=f.nullable)
    if pa.types.is_large_list(t):
        optimized_child = _optimize_field(t.field(0))
        return pa.field(f.name, pa.list_(optimized_child), nullable=f.nullable)
    if pa.types.is_dictionary(t):
        new_value_type = _optimize_field(pa.field("", t.value_type)).type
        return pa.field(f.name, pa.dictionary(pa.uint8(), new_value_type), nullable=f.nullable)
    return f

dataset_arrow = dataset.to_arrow()
optimized_schema = pa.schema([
    f if f.name in _KEEP_LARGE else _optimize_field(f)
    for f in dataset_arrow.schema
])
dataset_arrow = dataset_arrow.cast(optimized_schema)


_INSTRUCTION = """

The following is the schema for `dataset_arrow`.

```
id_file: string
id_utterance: uint64
speaker: string
morph_lemma: list<item: string>
  child 0, item: string
morph_metadata: large_list<item: large_string>
  child 0, item: large_string
morph_pos: list<item: dictionary<values=string, indices=uint8, ordered=0>>
  child 0, item: dictionary<values=string, indices=uint8, ordered=0>
morph_word_type: list<item: dictionary<values=string, indices=uint8, ordered=0>>
  child 0, item: dictionary<values=string, indices=uint8, ordered=0>
syntax_arc: list<item: uint64>
  child 0, item: uint64
syntax_rel: list<item: dictionary<values=string, indices=uint8, ordered=0>>
  child 0, item: dictionary<values=string, indices=uint8, ordered=0>
utterance: large_string
```


The following is a sample data. Notice that the `morph_metadata` field is a list of JSON strings, and the `utterance` field contains the raw text of the utterance.

```python
{'id_file': 'Eng-UK/Edinburgh/eve1510',
 'id_utterance': 11,
 'speaker': 'PAR0',
 'morph_lemma': ['look', 'it', 'be', 'a', 'bubble', 'iceberg', '.'],
 'morph_metadata': ['{"index_word": 1, "index_clitic": 1, "features": {"feature": [], "suffix": [], "prefix": [], "explanation": []}}',
  '{"index_word": 2, "index_clitic": 1, "subpos": "per", "features": {"feature": [], "suffix": [], "prefix": [], "explanation": []}}',
  '{"index_word": 2, "index_clitic": 2, "features": {"feature": ["3S"], "suffix": [], "prefix": [], "explanation": []}}',
  '{"index_word": 3, "index_clitic": 1, "subpos": "art", "features": {"feature": [], "suffix": [], "prefix": [], "explanation": []}}',
  '{"index_word": 4, "index_clitic": 1, "features": {"feature": [], "suffix": [], "prefix": [], "explanation": []}}',
  '{"index_word": 5, "index_clitic": 1, "components": [{"pos": "n", "lemma": "ice"}, {"pos": "n", "lemma": "berg"}], "features": {"feature": [], "suffix": [], "prefix": [], "explanation": []}}',
  '{"index_word": 6, "index_clitic": 1}'],
 'morph_pos': ['co', 'pro', 'cop', 'det', 'n', 'n', 'punct'],
 'morph_word_type': ['word',
  'word',
  'word',
  'word',
  'word',
  'compound',
  'punctuation'],
 'syntax_arc': [3, 3, 0, 6, 6, 3, 3],
 'syntax_rel': ['COM', 'SUBJ', 'ROOT', 'DET', 'MOD', 'PRED', 'PUNCT'],
 'utterance': "look it's a bubble iceberg ."}
```

"""