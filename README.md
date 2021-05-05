---
language: "en"
thumbnail: "https://pbs.twimg.com/profile_images/1092721745994440704/d6R-AHzj_400x400.jpg"
tags:
- propaganda
- bert
license: "MIT"
datasets:
-
metrics:
-
---

Propaganda Techniques Analysis BERT
----

This model is a BERT based model to make predictions of propaganda techniques in
news articles in English. The model is described in
[this paper](https://propaganda.qcri.org/papers/EMNLP_2019__Fine_Grained_Propaganda_Detection.pdf).

This repository contains a REST server and Docker container to perform the propaganda detection. The official API documented at https://app.swaggerhub.com/apis-docs/yifan2019/Tanbih/ is rate-limited so this repo provides the server.
This repository comes from https://huggingface.co/QCRI/PropagandaTechniquesAnalysis-en-BERT


## Model description

Please find propaganda definition here:
https://propaganda.qcri.org/annotations/definitions.html

You can also try the model in action here: https://www.tanbih.org/prta


### How to use

```python
>>> from transformers import BertTokenizerFast
>>> from .model import BertForTokenAndSequenceJointClassification
>>>
>>> tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
>>> model = BertForTokenAndSequenceJointClassification.from_pretrained(
>>>     "QCRI/PropagandaTechniquesAnalysis-en-BERT",
>>>     revision="v0.1.0",
>>> )
>>> 
>>> inputs = tokenizer.encode_plus("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)
>>> sequence_class_index = torch.argmax(outputs.sequence_logits, dim=-1)
>>> sequence_class = model.sequence_tags[sequence_class_index[0]]
>>> token_class_index = torch.argmax(outputs.token_logits, dim=-1)
>>> tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][1:-1])
>>> tags = [model.token_tags[i] for i in token_class_index[0].tolist()[1:-1]]
```


### BibTeX entry and citation info

```bibtex
@inproceedings{da-san-martino-etal-2019-fine,
    title = "Fine-Grained Analysis of Propaganda in News Article",
    author = "Da San Martino, Giovanni  and
      Yu, Seunghak  and
      Barr{\'o}n-Cede{\~n}o, Alberto  and
      Petrov, Rostislav  and
      Nakov, Preslav",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1565",
    doi = "10.18653/v1/D19-1565",
    pages = "5636--5646",
    abstract = "Propaganda aims at influencing people{'}s mindset with the purpose of advancing a specific agenda. Previous work has addressed propaganda detection at document level, typically labelling all articles from a propagandistic news outlet as propaganda. Such noisy gold labels inevitably affect the quality of any learning system trained on them. A further issue with most existing systems is the lack of explainability. To overcome these limitations, we propose a novel task: performing fine-grained analysis of texts by detecting all fragments that contain propaganda techniques as well as their type. In particular, we create a corpus of news articles manually annotated at fragment level with eighteen propaganda techniques and propose a suitable evaluation measure. We further design a novel multi-granularity neural network, and we show that it outperforms several strong BERT-based baselines.",
}
```
