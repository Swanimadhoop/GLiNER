import spacy
import gliner_spacy
from gliner_spacy.pipeline import GlinerSpacy
from spacy.language import Language
from spacy.tokens import Span
from gliner import GLiNER
import re
import warnings
import time

warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

# Configuration for GLiNER integration
custom_spacy_config = {
    "gliner_model": "small",
    "chunk_size": 250,
    "labels": ["PERSON","ORGANISATION","PHONE NUMBER", "EMAIL", "ADDRESS","DESIGNATION"],
    # "labels": ["DESIGNATION"],
    # "labels": ["ORGANISATION"],
    "style": "ent",
    "threshold": 0.1
}

# Initialize a blank English spaCy pipeline and add GLiNER
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
trained_model = GLiNER.from_pretrained("small", local_files_only=True)

# Define a custom component for email and URL detection
@Language.component("custom_ner_component")
def custom_ner_component(doc):
    # Check if "EMAIL" is in the entity list from the configuration
    if "EMAIL" in custom_spacy_config["labels"]:
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

start_time = time.time()

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



inference_time = time.time() - start_time

print("Inference time:", inference_time, "seconds")


# # Output detected entities
# for ent in doc.ents:
#     print(f"{ent.text} => {ent.label_}")
