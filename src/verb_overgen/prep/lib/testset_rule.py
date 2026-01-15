import unittest

import regex as re


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
    if re.search("o$|ch$|sh$|x$|s$|z$", verb):
        return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", r"\g<verb>es \g<object>", sentence_target)
    elif re.search(r"(?<![aeiou])y$", verb):
        return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", rf"{verb[:-1]}ies \g<object>", sentence_target)
    else:
        return re.sub(rf"makes (?P<object>.*) (?P<verb>{verb})", r"\g<verb>s \g<object>", sentence_target)


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


if __name__ == "__main__":
    unittest.main()
