# e2e_EL_multilingual_experiments

## run anyway

First, you need to install fasebookresearch/LASER.

- https://github.com/facebookresearch/LASER

And then, you also need to install laserencoder.

- https://github.com/sugiyamath/laserencoder

If you want to use ```predictor_ja.py```, you need to install MeCab and neologd.

- https://github.com/taku910/mecab
- https://github.com/neologd/mecab-ipadic-neologd

After that, you can download some data by running ```download_data.py```:

```
cd run_anyway
pip install googledrivedownloader
pip install -r requirements.txt
python download_data.py
cd modules
```

Finally, you can run the predictor:

```
python predictor.py  # or predictor_ja.py
```

```python
threshold>0.5
sent>Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aims to help programmers write clear, logical code for small and large-scale projects.
/tmp/tmp5hrr51xl
 - Tokenizer: tmp5hrr51xl in language en
 - fast BPE: processing tok
 - Encoder: bpe to QEITL4XGTU1T8F3Q
 - Encoder: 52 sentences in 0s
[{'entity': 'Python_(programming_language)', 'mention': 'Python', 'word_index': (0, 1)}, {'entity': 'Python_(programming_language)', 'mention': 'Python', 'word_index': (22, 23)}, {'entity': 'Programming_language', 'mention': 'programming language', 'word_index': (8, 10)}, {'entity': 'General-purpose_programming_language', 'mention': 'general-purpose programming language', 'word_index': (7, 10)}, {'entity': 'Guido_van_Rossum', 'mention': 'Guido van Rossum', 'word_index': (13, 16)}]
```

```python
threshold>0.2
sent>Django は Apache 2 で mod python を使って、あるいは任意の WSGI 準拠のウェブサーバで動作させることができる。NginxとuWSGIでも動作が可能となっている。 Django は FastCGI サーバを起動することができ、FastCGI をサポートする任意のウェブサーバのバックエンドで使用することができる。
/tmp/tmpy9rcjfab
 - Tokenizer: tmpy9rcjfab in language en
 - fast BPE: processing tok
 - Encoder: bpe to E54RO4XYCK303VZE
 - Encoder: 28 sentences in 0s
[{'entity': 'Django_(web_framework)', 'mention': 'Django', 'word_index': [(0, 1), (0, 1)]}, {'entity': 'Web_Server_Gateway_Interface', 'mention': 'WSGI', 'word_index': [(14, 15)]}, {'entity': 'Web_server', 'mention': 'ウェブサーバ', 'word_index': [(17, 18)]}, {'entity': 'Nginx', 'mention': 'Nginx', 'word_index': [(26, 27)]}, {'entity': 'UWSGI', 'mention': 'uWSGI', 'word_index': [(28, 29)]}, {'entity': 'Django_(web_framework)', 'mention': 'Django', 'word_index': [(39, 40), (39, 40)]}, {'entity': 'FastCGI', 'mention': 'FastCGI', 'word_index': [(41, 42)]}, {'entity': 'Server_(computing)', 'mention': 'サーバ', 'word_index': [(42, 43), (42, 43), (42, 43), (42, 43)]}, {'entity': 'FastCGI', 'mention': 'FastCGI', 'word_index': [(50, 51)]}, {'entity': 'Web_server', 'mention': 'ウェブサーバ', 'word_index': [(56, 57)]}, {'entity': 'Front_and_back_ends', 'mention': 'バックエンド', 'word_index': [(58, 59)]}]
```

The English predictor and the Japanese predictor use the same model, ```model_wiki_tmp.h5```. That's why the model is multilingual.