# e2e_EL_multilingual_experiments

## run anyway

First, you need to install fasebookresearch/LASER.

- https://github.com/facebookresearch/LASER

And then, you also need to install laserencoder.

- https://github.com/sugiyamath/laserencoder

If you want to use predictor_ja.py, you need to install MeCab and neologd.

- https://github.com/taku910/mecab
- https://github.com/neologd/mecab-ipadic-neologd

After that, you can download some data by running download_data.py:

```
cd run_anyway
pip install googledrivedownloader
pip install -r requirements.txt
python download_data.py
cd modules
```

Finally, you can run the predictors:

```
python predictor.py  # or predictor_ja.py
```

```python
threshold>0.5
sent>Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aims to help programmers write clear, logical code for small and large-scale projects.

[('Python_(programming_language)', 'Python'), ('Python_(programming_language)', 'Python'), ('Programming_language', 'programming language'), ('General-purpose_programming_language', 'general-purpose programming language'), ('Guido_van_Rossum', 'Guido van Rossum')]
```

```python
sent>Django は Apache 2 で mod python を使って、あるいは任意の WSGI 準拠のウェブサーバで動作させることができる。NginxとuWSGIでも動作が可能となっている。 Django は FastCGI サーバを起動することができ、FastCGI をサポートする任意のウェブサーバのバックエンドで使用することができる。
/tmp/tmplf7ubtnc
 - Tokenizer: tmplf7ubtnc in language en
 - fast BPE: processing tok
 - Encoder: bpe to I8YAA7XUHJ8QQJMK
 - Encoder: 28 sentences in 0s
[('Django_(web_framework)', 'Django'), ('Web_Server_Gateway_Interface', 'WSGI'), ('Web_server', 'ウェブサーバ'), ('Nginx', 'Nginx'), ('UWSGI', 'uWSGI'), ('Django_(web_framework)', 'Django'), ('FastCGI', 'FastCGI'), ('Server_(computing)', 'サーバ'), ('FastCGI', 'FastCGI'), ('Web_server', 'ウェブサーバ'), ('Front_and_back_ends', 'バックエンド')]
```

The English entity predictor and the Japanese entity predictor use same model, model_wiki_tmp.h5. That's why the model is called "multilingual EL model".