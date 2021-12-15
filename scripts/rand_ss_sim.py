import argparse
from pathlib import Path
from random_word import RandomWords
import random
from sentence_transformers import SentenceTransformer, util
import csv


random.seed(123)
sbert_model = SentenceTransformer("whaleloops/phrase-bert")

CORONA_NAME_VARIATIONS = [
    "COVID-19",
    "Coronavirus",
    "SARS-CoV-2",
    "coronavirus",
    "corona virus",
    "covid-19",
    "covid 19",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, help="Length of sentences to create.")
    args = p.parse_args()

    r = RandomWords()

    topics_info = {}
    with open(
        Path(__file__).parent.parent / "cdse_covid" / "claim_detection" / "topic_list.txt",
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            key = row[3]
            topics_info[key] = {"topic": row[2], "subtopic": row[1]}

    templates = list(topics_info.keys())

    def clean_templates(dirty_templates, replacement_word="COVID-19"):
        cleaned_templates = []
        for sentence in dirty_templates:
            for name in CORONA_NAME_VARIATIONS:
                sentence = sentence.replace(name, replacement_word)
            sentence = sentence.replace("X", "this")
            cleaned_templates.append(sentence)
        return cleaned_templates

    sim_num = 100
    all_r_sentences = []
    count = 0
    while count != sim_num:
        random_words = r.get_random_words(hasDictionaryDef=True, limit=args.length)
        if random_words:
            placement_idx = random.randint(0, 3)
            random_words.insert(placement_idx, "COVID-19")
            rw_as_sentence = " ".join(random_words)
            all_r_sentences.append(rw_as_sentence)
            count += 1
    encoded_r_sentences = sbert_model.encode(all_r_sentences, convert_to_tensor=True)

    covid_templates = clean_templates(templates)
    encoded_covid_templates = sbert_model.encode(covid_templates, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(encoded_covid_templates, encoded_r_sentences)
    averages_for_templates_with_covid = []
    for template in similarity_matrix:
        avg_for_template = template.mean()
        averages_for_templates_with_covid.append(avg_for_template)

    templates = clean_templates(templates, "")
    encoded_templates = sbert_model.encode(templates, convert_to_tensor=True)
    sim_matrix_2 = util.pytorch_cos_sim(encoded_templates, encoded_r_sentences)
    averages_for_templates = []
    for template in sim_matrix_2:
        avg_for_template = template.mean()
        averages_for_templates.append(avg_for_template)

    with open("phrase_out0.csv", "w+") as handle:
        writer = csv.writer(handle)
        for i, (template_covid, template_no_covid) in enumerate(
            zip(averages_for_templates_with_covid, averages_for_templates)
        ):
            writer.writerow([covid_templates[i], float(template_covid), float(template_no_covid)])


if __name__ == "__main__":
    main()
