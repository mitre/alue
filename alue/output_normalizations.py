import re
import string

TASK_SPECIFIC_PATTERNS = {
    "pirep_classification": {
        "EXACT_OUTPUT_PATTERN": "^[01]$",
        "SEARCH_PATTERN": "^([01])",
        "no_match_return_value": "-1",
    },
    "atc_dialogue_classification": {
        "EXACT_OUTPUT_PATTERN": r"^\[[01],[01],[01]\]$",
        "SEARCH_PATTERN": r"(\[[01], *[01], *[01]\])",
        "no_match_return_value": "[0,0,0]",
    },
    "ntsb_damage_classification": {
        "EXACT_OUTPUT_PATTERN": r'\["([A-Z]+)"\]',
        "SEARCH_PATTERN": "^([A-Z]+)",
        "no_match_return_value": "NA",
    },
    "ntsb_damage_multilabel_classification": {
        "EXACT_OUTPUT_PATTERN": "^[A-Z]+(, *[A-Z]+)*$",
        "SEARCH_PATTERN": "^([A-Z]+(, *[A-Z]+)*)",
        "no_match_return_value": "NA",
        "label_names": ["NONE", "MINR", "SUBS", "DEST"],
    },
    "asrs_factors_multilabel_classification": {
        "EXACT_OUTPUT_PATTERN": r"^\[?['a-zA-Z /\(\)-]+(, *['a-zA-Z /\(\)-]+)*\]?$",
        "SEARCH_PATTERN": r"^\[?(['a-zA-Z /\(\)-]+(, *['a-zA-Z /\(\)-]+)*)\]?",
        "no_match_return_value": "NA",
        "label_names": [
            "Manuals",
            "Procedure",
            "Human Factors",
            "Airspace Structure",
            "Chart Or Publication",
            "Aircraft",
            "Environment - Non Weather Related",
            "Airport",
            "Software and Automation",
            "Incorrect / Not Installed / Unavailable Part",
            "Staffing",
            "Weather",
            "Logbook Entry",
            "ATC Equipment / Nav Facility / Buildings",
            "Company Policy",
            "Equipment / Tooling",
            "Minimum Equipment List",
            "None",
        ],
    },
}


def normalize_schema_output(
    task_normalization_name,
    data_dict,
):
    if "ntsb_damage" in task_normalization_name:
        # Output dictionary
        output_data = {}

        # Iterate through the input data
        for key, value in data_dict.items():
            # Extract all damage types from the damages list
            damage_types = []
            for entry in value:
                print(f"entry: {entry}")
                for damage in entry.get("damages", []):
                    damage_types.append(damage["damage_type"])

            # Normalize the damage types into a single string (comma-separated)
            normalized_damage_types = ", ".join(sorted(set(damage_types)))
            output_data[key] = normalized_damage_types

        return output_data

    elif "asrs_factors" in task_normalization_name:
        output_data = {}
        for qa_id, qa_output in data_dict.items():
            normalized_labels = ", ".join(sorted(set(qa_output[0]["labels"])))
            output_data[qa_id] = normalized_labels

        return output_data


def normalize_sequence_classification_generation_output(
    exact_output_pattern: re.Pattern,
    search_pattern: re.Pattern,
    text: str,
    exact_match_only: bool = False,
    no_match_return_value: str = "-1",
):
    text = text.strip()
    # look for exact match only
    if exact_output_pattern.match(text):
        return text
    elif not exact_match_only:
        match_output = search_pattern.match(text)
        if match_output:
            print(f"Found match: {match_output.group(1)}")
            return match_output.group(1)

    print(f"No match in: {text}. Returning {no_match_return_value}.")
    return no_match_return_value


def normalize_ner_generation_output(raw_output):
    normalized_output = []
    # split raw generation output on new lines
    lines = [line.strip() for line in raw_output.split("\n")]
    for _i, line in enumerate(lines):
        # keep only lines that contain a colon
        if ":" in line:
            normalized_output.append(line)
    # print('\n'.join(normalized_output))
    return "\n".join(normalized_output)


def find_phrase_indices_in_text(phrase, text):
    # create empty return list
    return_word_indices = []
    # convert strings to lists of words
    phrase_word_list = phrase.split()
    text_word_list = text.split()
    nbr_phrase_words = len(phrase_word_list)
    # only search if phrase contains words
    if nbr_phrase_words > 0:
        # iterate over words in text and look for spans that match phrase word list
        for j in (
            i for i, word in enumerate(text_word_list) if word == phrase_word_list[0]
        ):
            print(j)
            if text_word_list[j : j + nbr_phrase_words] == phrase_word_list:
                # return [starting index, ending index)
                return_word_indices.append((j, j + nbr_phrase_words))
    # return list of indices tuples
    return return_word_indices


def transform_ner_generation_output_to_token_classification_output(
    text_input, prediction
):
    # create raw output
    output = ["0"] * len(text_input.split(" "))
    # covert prediction to list of tuples containing label and phrase
    normalized_prediction = normalize_ner_generation_output(prediction)
    # print(normalized_prediction)
    if len(normalized_prediction.strip()) > 0:
        predicted_entities = [
            (s.split(":")[0].strip(), s.split(":")[1].strip())
            for s in normalized_prediction.split("\n")
        ]
        # print(predicted_entities)
        # iterate over predicted entities
        for label, phrase in predicted_entities:
            # find word indices
            print("*****")
            print(phrase, " // ", text_input)
            print("-----")
            matched_spans = find_phrase_indices_in_text(phrase, text_input)
            # print(matched_spans)
            # replace all matched word indices with label
            for start, end in matched_spans:
                output = [
                    label if i in range(start, end) else v for i, v in enumerate(output)
                ]
    # return output
    print(" ".join(output))
    return " ".join(output)


def extract_markup_tags_and_text(sentence):
    """
    Looks for markup tags that start with the <tag> and end with a matching </tag> within a given
    sentence. It extracts the "tag" name and text within the tags.

    params:
    sentence: the sentence where to search for markup tags

    Returns
    -------
    tag_names: a list of the names of tags found within the sentence
    tag_texts: a list of texts that was found within the tags, corresponding to the tag_names list.
    """
    # Regular expression to match markup tags and their content
    pattern = r"<(\w+)>(.*?)</\1>"
    # Find all matches in the sentence
    matches = re.findall(pattern, sentence)
    # Separate matches into two lists: one for tag names and one for text inside the tags
    tag_names = [tag for tag, _ in matches]
    tag_texts = [text for _, text in matches]
    return tag_names, tag_texts


# Function to remove punctuation from a string
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def find_substring_indexes(sentence, substring):
    """
    Searches for a substring span within a sentence and if found, returns the index range
    where the span exists within the sentence.

    params:
    sentence: the sentece where to search for a given substring
    substring: the substring span

    Returns
    -------
    a list of indexes spanning the sentence where the substring exists
    """
    # Remove punctuation from both the sentence and substring
    sentence_cleaned = remove_punctuation(sentence)
    substring_cleaned = remove_punctuation(substring)

    # Split the cleaned sentence and substring into words
    sentence_words = sentence_cleaned.split()
    substring_words = substring_cleaned.split()

    # Get the length of the substring in words
    substring_length = len(substring_words)
    # Iterate through the sentence to find the matching span
    for i in range(len(sentence_words) - substring_length + 1):
        # Check if the words match the substring
        if sentence_words[i : i + substring_length] == substring_words:
            # Return the indexes of the matching span
            return list(range(i, i + substring_length))

    # If no match is found, return an empty list
    return []
