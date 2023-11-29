Repository for quantum correlations investigation with the usage of neural networks.

Mandatory packages are listed in requirements.txt file. To install them run:
```
pip3 install -r requirements.txt
```

To generate datasets run:
```
python3 run/data/generate_datasets.py
```
with parameters of your choice specified inside the file (e.g. for paper "IdentifiIdentification of quantum entanglement with Siamese convolutional neural networks and semi-supervised learning", set flag paper ='entanglement', and for "Data-driven criteria for quantum correlations" set paper = 'discord')


To evaluate the models run:

For CNN and Siamese CNN from entanglement paper:
```
python3 run/test_bipart_classifier.py
```
with siamese_flag set appropriately. If evaluating on verified dataset set verified_dataset = True, if evaluating on negativity labeled dataset set it to False.

For ensemble model from entanglement paper:
```
python3 run/test_multi_bipart_classifier.py
```
with verified_dataset flag set adequately.

For Separator model from discord paper:
```
python3 run/test_separator.py
```


To train the models use scripts:
- entanglement: run/train_bipart_classifier.py and run/train_multi_bipart_classifier.py with parameters set appropriately.
- discord: run/train_separator.py with parameters set appropriately.


Scripts for additional plots can be found in entanglement/plots and discord/plots for respective papers.
