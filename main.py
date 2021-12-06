import json
from field_extraction import *

resume_text = extract_text1('./resumes/Yashwik panchal RESUME.pdf')
# print(resume_text)
text = ' '.join(resume_text.split())

data = {}
data['name'] = stanfordNER_name(resume_text)
data['phone_no'] = extract_mobile_number(text)
data['email'] = extract_email(resume_text)
data['links'] = extract_links(resume_text)
data['skills'] = extract_skills(resume_text)
data['education'] = extract_education(resume_text)
entities = extract_entity_sections(resume_text)

college = ' '.join([str(item) for item in entities['education']])
data['college & schools'] = stanfordNER_college(college)

if 'experience' in entities:
    data['experience'] = entities['experience']
else:
    data['experience'] = []

data['entities'] = entities
data['competencies'] = extract_competencies(resume_text)

json_obj = json.dumps(data, indent=4)
with open("sample.json", "w") as outfile:
    outfile.write(json_obj)
