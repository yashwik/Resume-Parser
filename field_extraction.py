import re
import io
import os
import pandas
import docx2txt
import spacy
import nltk
import constants as cs
from nltk.tree import Tree
from nltk import word_tokenize, pos_tag, ne_chunk
from spacy.matcher import Matcher
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

# load pre-trained model
nlp = spacy.load('en_core_web_sm')

# Education (Upper Case Mandatory)
EDUCATION         = [
                    'BE','B.E.', 'B.E', 'BS', 'B.S', 'ME', 'M.E', 'M.E.', 'MS', 'M.S', 'BTECH', 'MTECH',
                    'SSC', 'HSC', 'CBSE', 'ICSE', 'B-TECH', 'Bachelor of Science', 'Bachelor of Technology'
                    ]

NOT_ALPHA_NUMERIC = r'[^a-zA-Z\d]'

NUMBER            = r'\d+'

# For finding date ranges
MONTHS_SHORT      = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
MONTHS_LONG       = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
MONTH             = r'(' + MONTHS_SHORT + r'|' + MONTHS_LONG + r')'
YEAR              = r'(((20|19)(\d{2})))'

STOPWORDS         = set(stopwords.words('english'))

RESUME_SECTIONS = [
                    'accomplishments',
                    'experience',
                    'education',
                    'interests',
                    'projects',
                    'professional experience',
                    'publications',
                    'skills',
                    'activities'
                ]

RESERVED_WORDS = [
    'school',
    'college',
    'univers',
    'academy',
    'faculty',
    'institute',
    'faculdades',
    'Schola',
    'schule',
    'lise',
    'lyceum',
    'lycee',
    'polytechnic',
    'kolej',
    'ünivers',
    'okul',
]
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()

            # create a file handle
            fake_file_handle = io.StringIO()

            # creating a text converter object
            converter = TextConverter(
                resource_manager,
                fake_file_handle,
                codec='utf-8',
                laparams=LAParams()
            )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                resource_manager,
                converter
            )

            # process current page
            page_interpreter.process_page(page)

            # extract text
            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()

def extract_text_from_doc(doc_path):
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)

def extract_text(file_path, extension):
    text = ''
    if extension == '.pdf':
        for page in extract_text_from_pdf(file_path):
            text += ' ' + page
    elif extension == '.docx' or extension == '.doc':
        text = extract_text_from_doc(file_path)
    return text

def extract_name(resume_text):
    nlp_text = nlp(resume_text)

    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

    matcher.add('NAME', [pattern])

    matches = matcher(nlp_text)

    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text

def extract_names(txt):
    person_names = []

    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )

    return person_names

def extract_mobile_number(text):
    phone = re.findall(re.compile(
        r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{5})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?|(\([0-9]{3}\) [0-9]{3}-[0-9]{4})'),text)

    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return '+' + number
        else:
            return number

def extract_email(text):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None

def extract_links(text):
    git = re.findall(re.compile(
        r"(?:https//?:)?(?:www\.?:)?github\.com/(?:[A-z0-9_-]+)/?"),text)
    linkedin = re.findall(re.compile(
        r"(?:https//?:)?(?:[\w]+\.)?linkedin\.com/in/(?:[\w\-_À-ÿ%]+)/?"), text)

    links={}

    if git:
        github = ''.join(git[0])
        links['github'] = github
    else:
        links['github'] = None

    if linkedin:
        link = ''.join(linkedin[0])
        links['linkedin'] = linkedin
    else:
        links['linkedin'] = None

    return links


def get_continuous_chunks(text, chunk_func=ne_chunk):
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

def extract_skills(resume_text):
    '''
    Helper function to extract skills from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param noun_chunks: noun chunks extracted from nlp text
    :return: list of skills extracted
    '''
    nlpt = ' '.join(resume_text.split())
    nlp_text = nlp(nlpt)
    noun_chunks = list(nlp_text.noun_chunks)
    nounc = get_continuous_chunks(resume_text)

    tokens = [token.text for token in nlp_text if not token.is_stop]
    data = pandas.read_csv(os.path.join(os.path.dirname(__file__), 'skills.csv'))
    skills = list(data.columns.values)
    skillset = []
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    for token in nounc:
        token = token.lower().strip()
        if token in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def extract_education(resume_text):
    '''
    Helper function to extract education from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :return: tuple of education degree and year if year if found else only returns education degree
    '''
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            tex = re.sub(r'[?|$|.|!|,|(|)]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                    edu[tex] = text + nlp_text[index]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(YEAR), edu[key])
        if year:
            education.append((key, ''.join(year.group(0))))
        else:
            education.append(key)
    return education


def extract_entity_sections(text):
    '''
    Helper function to extract all the raw text from sections of resume

    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    text_split = [i.strip() for i in text.split('\n')]
    # sections_in_resume = [i for i in text_split if i.lower() in sections]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(RESUME_SECTIONS)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in RESUME_SECTIONS:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)

    # entity_key = False
    # for entity in entities.keys():
    #     sub_entities = {}
    #     for entry in entities[entity]:
    #         if u'\u2022' not in entry:
    #             sub_entities[entry] = []
    #             entity_key = entry
    #         elif entity_key:
    #             sub_entities[entity_key].append(entry)
    #     entities[entity] = sub_entities

    # pprint.pprint(entities)

    # make entities that are not found None
    # for entity in cs.RESUME_SECTIONS:
    #     if entity not in entities.keys():
    #         entities[entity] = None
    return entities


def extract_experience(resume_text):
    '''
    Helper function to extract experience from resume text

    :param resume_text: Plain resume text
    :return: list of experience
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize
    filtered_sentence = [w for w in word_tokens if
                         not w in stop_words and wordnet_lemmatizer.lemmatize(w) not in stop_words]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)

    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)

    test = []

    for vp in list(cs.subtrees(filter=lambda x: x.label() == 'P')):
        test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))

    # Search the word 'experience' in the chunk and then print out the text after it
    x = [x[x.lower().index('experience') + 10:] for i, x in enumerate(test) if x and 'experience' in x.lower()]
    return x


def extract_competencies(text):
    '''
    Helper function to extract competencies from resume text

    :param resume_text: Plain resume text
    :return: dictionary of competencies
    '''
    competency_dict = {}

    for competency in cs.COMPETENCIES.keys():
        for item in cs.COMPETENCIES[competency]:
            if string_found(item, text):
                if competency not in competency_dict.keys():
                    competency_dict[competency] = [item]
                else:
                    competency_dict[competency].append(item)

    return competency_dict


def extract_measurable_results(text, experience_list):
    '''
    Helper function to extract measurable results from resume text

    :param resume_text: Plain resume text
    :return: dictionary of measurable results
    '''

    # we scan for measurable results only in first half of each sentence
    experience_text = ' '.join([text[:len(text) // 2 - 1] for text in experience_list])
    mr_dict = {}

    for mr in cs.MEASURABLE_RESULTS.keys():
        for item in cs.MEASURABLE_RESULTS[mr]:
            if string_found(item, experience_text):
                if mr not in mr_dict.keys():
                    mr_dict[mr] = [item]
                else:
                    mr_dict[mr].append(item)

    return mr_dict


def string_found(string1, string2):
    if re.search(r"\b" + re.escape(string1) + r"\b", string2):
        return True
    return False