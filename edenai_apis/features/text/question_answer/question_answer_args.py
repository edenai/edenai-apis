# pylint: disable=locally-disabled, line-too-long
question_answer_sample = {
    "texts": [
        "The bar-shouldered dove (Geopelia humeralis) is a species of dove, in the family Columbidae, native to Australia and southern New Guinea. Its typical habitat consists of areas of thick vegetation where water is present, damp gullies, forests and gorges, mangroves, plantations, swamps, eucalyptus woodland, tropical and sub-tropical shrubland, and river margins. It can be found in both inland and coastal regions."
    ],
    "question": "What is the scientific name of bar-shouldered dove?",
    "temperature": 0,
    "examples_context": "In 2017, U.S. life expectancy was 78.6 years.",
    "examples": [["What is human life expectancy in the United States?", "78 years."]],
    "model": None,
}


def question_answer_arguments():
    return {
        "texts": question_answer_sample["texts"],
        "question": question_answer_sample["question"],
        "temperature": question_answer_sample["temperature"],
        "examples_context": question_answer_sample["examples_context"],
        "examples": question_answer_sample["examples"],
        "model": None,
    }
