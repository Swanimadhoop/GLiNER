import spacy
import json
import warnings


warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

# train_path = "Sample_data1.json"
train_path = "Sample_Data.json"


with open(train_path, "r") as f:
    data = json.load(f)

import torch
from gliner import GLiNER

# If you want to use fp16 training, you need to install the following packages
# install accelerate and beartype if not already done
from trainer import GlinerTrainer
 
model = GLiNER.from_pretrained("EmergentMethods/gliner_medium_news-v2.1")
# model = GLiNER.from_pretrained("urchade/gliner_small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = model.to(device)
eval_data = {
    "entity_types": ["DESIGNAION","ORGANISATION"],
    # "entity_types": ["PERSON","ORGANISATION","PHONE NUMBER", "EMAIL", "ADDRESS","DESIGNATION"],

    "samples": data[:50]
}

trainer = GlinerTrainer(model, 
                        train_data = data[50:],
                        batch_size = 4,
                        grad_accum_every = 16,
                        lr_encoder = 1e-5,
                        lr_others = 5e-5, 
                        freeze_token_rep = False, 
                        val_every_step = 1000, 
                        val_data = eval_data,
                        checkpoint_every_epoch = 2, # Or checkpoint_every_step if you use steps
                        max_types=25,
                        ## ... add more
                        #Uncomment these if you want to train using fp16 
                        #optimizer_kwargs = { "eps": 1e-7},  # Using higher eps might cause NaN loss
                        #accelerator_kwargs = {"mixed_precision" : "fp16" },
)


trainer.train(num_epochs=50) # Or by steps: trainer.train(num_steps=50)
output_dir = "final"
trainer.model.save_pretrained(output_dir)
md = GLiNER.from_pretrained("final", local_files_only=True)


# text = """
# CVinodhkumar
# CTO
# +919744172744
# Muthoottu
# vinodhkumar.c@muthoottumini.com
# www.muthoottumini.com
# Muthoottu Royal Tower, Kaloor
# Cochin, Kerala-682017
# """

# # Labels for entity prediction
# labels = ["ORGANISATION","DESIGNATION"] # for v2.1 use capital case for better performance

# # Perform entity prediction
# entities = model.predict_entities(text, labels, threshold=0.5)

# # Display predicted entities and their labels
# for entity in entities:
#     print(entity["text"], "=>", entity["label"])

 