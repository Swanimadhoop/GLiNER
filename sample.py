import spacy
from gliner_spacy.pipeline import GlinerSpacy
from spacy.language import Language
from spacy.tokens import Span
import re

# Configuration for GLiNER integration
custom_spacy_config = {
    "gliner_model": "EmergentMethods/gliner_medium_news-v2.1",
    "chunk_size": 250,
    "labels": ["PERSON", "ORGANISATION", "EMAIL", "ADDRESS", "PHONE NUMBER","PIN NUMBER"],
    "style": "ent",
    "threshold": 0.3
}

# Initialize a blank English spaCy pipeline and add GLiNER
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

# Define a custom component for email and URL detection
@Language.component("custom_ner_component")
def custom_ner_component(doc):
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    url_pattern = re.compile(r'\b(?:https?://|www\.)\S+\b')

    # Find all matches in the text
    email_matches = [(m.start(), m.end(), "EMAIL") for m in email_pattern.finditer(doc.text)]
    url_matches = [(m.start(), m.end(), "URL") for m in url_pattern.finditer(doc.text)]

    # Create spans for the matches
    spans = [Span(doc, doc.char_span(start, end).start, doc.char_span(start, end).end, label=label) for start, end, label in email_matches + url_matches]

    # Filter out spans that might be incorrectly labeled by GLiNER
    filtered_spans = [span for span in doc.ents if span.label_ != "EMAIL"] + spans
    doc.ents = filtered_spans
    return doc

# Add the custom component to the pipeline
nlp.add_pipe("custom_ner_component", after="gliner_spacy")

# Example text for entity detection
text = """
HIGH
3inba
as
ACHIEVERS
CLUBFY2023-24
Equitas SmallFinance Bank
Shanmugam Manivannan
National Manager
Equitas Small Finance Bank Limited
+919962510100
401A,IVFloor,Spencer Plaza,Phasel,
shanmugamm2@equitasbank.com
769 Anna Salai,Chennai-600002
www.equitasbank.com
"""

# Process the text with the pipeline
doc = nlp(text)

# Consolidate consecutive address tokens into single entity
consolidated_entities = []
current_address = []

for ent in doc.ents:
    if ent.label_ == "ADDRESS":
        # Collect consecutive address tokens
        current_address.append(ent.text)
    else:
        if current_address:
            # If we have collected address parts, consolidate them
            consolidated_entities.append({"text": ", ".join(current_address), "label": "ADDRESS"})
            current_address = []  # Clear for the next address
        # Add the non-address entity
        consolidated_entities.append({"text": ent.text, "label": ent.label_})

# Add the last collected address if any
if current_address:
    consolidated_entities.append({"text": ", ".join(current_address), "label": "ADDRESS"})

# Output detected entities, ensuring each is printed on a new line
for entity in consolidated_entities:
    print(f"{entity['text']} => {entity['label']}")

# # Output detected entities
# for ent in doc.ents:
#     print(f"{ent.text} => {ent.label_}")
