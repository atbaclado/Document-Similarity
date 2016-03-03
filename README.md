# Context-based similar documents
---

# 1. Introduction
In this repository, I implemented Context-based similar documents according to [Rich Anchor's blog](http://blog.richanchor.com/2016/01/22/context-based-similar-documents/)

Context-based similar documents is a problem that finding the most similar documents of the given document. The blog's approach is using LDA (Latent Dirichlet Allocation) Model to build the generic topics of the documents in database, then vectorize them, and using LSH (Locality Sensitive Hashing) to find the most similar documents (nearest neighbors) of the given document.

This software is written by Python 2.x with LDA Model provided by [gensim](https://radimrehurek.com/gensim/) and LSHForest provided by [scikit-learn](http://scikit-learn.org/)

# 2. Installation

This software depends on NumPy, Scikit-learn, Gensim - Python packages for scientific computing. You must have them installed prior to using vnSRL.

The simple way to install them is using pip:

```sh
	# pip install -U numpy scikit-learn gensim
```

# 3. Usage

## 3.1 Data 

This software requires 2 data files for training:

- Input data file includes documents in database. Each document is written on each line.
- Stop words file includes stop words which are are filtered out before  processing data.

And for query, we need 1 file stored given documents.

To run demo, can download sample input files. Those are extracted from [eva.vn](http://eva.vn/) provided by [Rich Anchor Team](https://richanchor.com/):

- [Input data](https://drive.google.com/file/d/0Byl51yNZoDkWMFhmR0VSVVpTbzA/view?usp=sharing)
- [Stop words](https://drive.google.com/file/d/0Byl51yNZoDkWamVOcEtZQm0zc0E/view?usp=sharing)

# 3.2 Quick-start

You can use this software by the following command-line:

```sh
python main.py
```

You can modify source code to fit your data:

- **[main.py](https://github.com/khoaipx/Document-Similarity/blob/master/Document-Similarity/main.py)**: 
    * ``input_file``: path to input data file
    * ``stopwords_file``: path to stop words file
    * ``num_topics``: number of topics in LDA Model
    * ``prefix_name``: prefix name of saved files (dictionary, corpus, model, etc.)
    * ``directory``: path to saved data directory
    * ``query``: path to query file
- **[setting.py](https://github.com/khoaipx/Document-Similarity/blob/master/Document-Similarity/setting.py)**: stores default setting
- **[corpus.py](https://github.com/khoaipx/Document-Similarity/blob/master/Document-Similarity/corpus.py)**:
    * ``get_docs()**: modify to fit data format

# 4. References

- Rich Anchor Blog [Context-based similar documents](http://blog.richanchor.com/2016/01/22/context-based-similar-documents/)
- Rich Anchor Blog [Topic modeling with LDA](http://blog.richanchor.com/2016/01/15/topic-modeling-with-lda/)
- Gensim [API Tutorial](https://radimrehurek.com/gensim/apiref.html)
- scikit-learn [User Guide](http://scikit-learn.org/stable/user_guide.html)

# 5. Contact

Xuan-Khoai Pham <phamxuankhoai@gmail.com>
