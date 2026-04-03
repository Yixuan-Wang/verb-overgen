import hashlib
import unittest

import polars as pl
import regex as re

from loguru import logger

def alternate_transitive(sentence_target: str, verb: str) -> str:
    """
    Convert a target sentence (the premmpting construction, make O V) to use the
    transitive construction (V O).

    The construction is made into 3-person singular by default.
    Since we are lemmatizing the sentence this won't have effect in the resulted token.


    Args:
        sentence_target (str): The original target sentence.
        verb (str): The verb to be transformed.
    """
    if not isinstance(sentence_target, str):
        logger.error(f"Expected sentence_target to be a string, got {type(sentence_target)} when verb is {verb}")
        return ""

    if "makes" in sentence_target:
        if re.search("o$|ch$|sh$|x$|s$|z$", verb):
            return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", r"\g<verb>es \g<object>", sentence_target)
        elif re.search(r"(?<![aeiou])y$", verb):
            return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", rf"{verb[:-1]}ies \g<object>", sentence_target)
        else:
            return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", r"\g<verb>s \g<object>", sentence_target)
    elif "make" in sentence_target:
        return re.sub(rf"make (?P<object>.*) (?P<verb>{verb})", r"\g<verb> \g<object>", sentence_target)
    else:
        raise ValueError(f"Sentence does not contain 'make' or 'makes': {sentence_target}")


def sha256_hash(*expr: pl.Expr) -> pl.Expr:
    """Hash the given expressions using SHA256 and return the hash as a hexadecimal string."""
    concatenated = pl.concat_str(expr, separator="\t")
    hash_expr = concatenated.map_elements(lambda x: hashlib.sha256(x.encode()).hexdigest(), pl.String)
    return hash_expr


def extract_causative_sentences(
    output_text: pl.Expr,
    *,
    verb: pl.Expr,
):
    """
    Examples:
    ```
    dataset.select(
        extract_sentences(
            pl.col("llm_gen").struct.field("output_text"),
            verb=pl.col("verb")
        )
    )
    ```
    """
    output_text = output_text.str.split("\n")
    sentence_context = output_text.list.first().str.strip_chars().alias("context")
    sentence_causative = output_text.list.get(1).str.strip_chars().alias("causative")
    sentence_transitive = (
        pl.struct(sentence_causative, verb)
        .map_elements(lambda s: alternate_transitive(s["causative"], s["verb"]), pl.String)
        .alias("transitive")
    )
    return pl.struct(sentence_context, sentence_causative, sentence_transitive).alias("sentence")


class TestStringMethods(unittest.TestCase):
    def test_alternate_transitive(self):
        self.assertEqual(
            alternate_transitive("Tom makes the cat play", "play"),
            "Tom plays the cat",
        )
        self.assertEqual(
            alternate_transitive("Tom makes the baby cry", "cry"),
            "Tom cries the baby",
        )
        self.assertEqual(
            alternate_transitive("Tom makes the boy laugh", verb="laugh"),
            "Tom laughs the boy",
        )
        self.assertEqual(
            alternate_transitive("Tom makes Jerry smile", verb="smile"),
            "Tom smiles Jerry",
        )
        self.assertEqual(
            alternate_transitive("We make the baby watch", verb="watch"),
            "We watch the baby",
        )


if __name__ == "__main__":
    unittest.main()
