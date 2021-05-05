from transformers import BertTokenizerFast
from model import BertForTokenAndSequenceJointClassification

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenAndSequenceJointClassification.from_pretrained(
    "QCRI/PropagandaTechniquesAnalysis-en-BERT",
    revision="v0.1.0",
)