# ReFT

Here is a small implementation of ReFT to go along with our weekly Arxiv Dive.

# Training / Test

```bash
# Download the minimal training data
oxen download ox/oxen-qa train.jsonl
# Train the model
python train.py
# Test the model
python -i test.py
```

# Acknowledgements

Thanks to the hard work and references from

* https://github.com/stanfordnlp/pyreft
* https://github.com/nicknochnack/PyReft