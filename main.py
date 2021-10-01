import json
from field_extraction import *
from pdfminer.high_level import extract_text
from nltk.tag.stanford import StanfordNERTagger
from nltk import pos_tag
from nltk.chunk import conlltags2tree
PATH_TO_JAR='C:/Users/ASUS/Downloads/stanford-ner-4.2.0/stanford-ner-2020-11-17/stanford-ner.jar'
PATH_TO_MODEL = 'C:/Users/ASUS/Downloads/stanford-ner-4.2.0/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz'
tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL,path_to_jar=PATH_TO_JAR, encoding='utf-8')

resume_text = extract_text('./Saurav_Kanegaonkar_v1.pdf','.pdf')
words = nltk.word_tokenize(resume_text)
ne_tagged_sent = tagger.tag(words)
# def get_continuous_chunks(tagged_sent):
#     continuous_chunk = []
#     current_chunk = []
#
#     for token, tag in tagged_sent:
#         if tag != "O":
#             current_chunk.append((token, tag))
#         else:
#             if current_chunk: # if the current chunk is not empty
#                 continuous_chunk.append(current_chunk)
#                 current_chunk = []
#     # Flush the final current_chunk into the continuous_chunk, if any.
#     if current_chunk:
#         continuous_chunk.append(current_chunk)
#     return continuous_chunk
#
#
# named_entities = get_continuous_chunks(ne_tagged_sent)
# named_entities = get_continuous_chunks(ne_tagged_sent)
# named_entities_str = [" ".join([token for token, tag in ne]) for ne in named_entities]
# named_entities_str_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities]
#
# print(named_entities_str_tag)

def stanfordNE2BIO(tagged_sent):
    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged_sent.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag

    return bio_tagged_sent


def stanfordNE2tree(ne_tagged_sent):
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = conlltags2tree(sent_conlltags)
    return ne_tree

ne_tree = stanfordNE2tree(ne_tagged_sent)

ne_in_sent = []
for subtree in ne_tree:
    if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
        ne_label = subtree.label()
        ne_string = " ".join([token for token, pos in subtree.leaves()])
        ne_in_sent.append((ne_string, ne_label))

print(ne_in_sent)
#------------------------------------------------------------------

data = {}
for word in ne_in_sent:
    if(word[1] == 'PERSON'):
        data['name'] = word[0]
        break

text = ' '.join(resume_text.split())
data['phone_no'] = extract_mobile_number(text)
data['email'] = extract_email(resume_text)
data['links'] = extract_links(resume_text)
data['skills'] = extract_skills(resume_text)
data['education'] = extract_education(resume_text)
data['experience'] = extract_experience(text)
data['entities'] = extract_entity_sections(resume_text)
entities = extract_entity_sections(resume_text)
data['competencies'] = extract_competencies(resume_text)

json_obj = json.dumps(data, indent=4)
with open("sample.json", "w") as outfile:
    outfile.write(json_obj)
