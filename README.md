Dit is het README.md-bestand van een project van het datalab van OCW. 

Pipeline:
01_datapreparatie bevat de code om de dataset samen te stellen.

- inladen_zoekwoorden.R en inlezen_pdf_bestanden.R bevatten code om de documenten van de API van de Tweede Kamer op te halen en vervolgens te filteren op zoekwoorden Onderwijs, Cultuur of Wetenschap in de metadata. Vervolgens worden de pdf-bestanden omgezet in tekst en wordt er gefilterd op aanwezigheid van tenminste één zoekwoord uit de lijst. Deze code is ontwikkeld door Coen Eisma (https://www.coeneisma.nl/), senior data-analist en coördinator van het Datalab bij het Ministerie van Onderwijs, Cultuur en Wetenschap.

- split_annotated_data_test.ipynb en merge_annotated_data_test.ipynb bevatten de code om de dataset te splitten en mergen voor en na het labellen.
- build_dataset_dummy.ipynb en build_dataset_test.ipynb bevatten wat statistieken over de dataset.
- build_inter_annotator_data.ipynb bevat de split van de data om te zorgen dat er genoeg data kan worden gelabeld om inter-annotator agreement over te berekenen.
- inter_annotator_agreement.ipynb bevat code om de inter-annotator agreement van een sample gelabelde data te berekenen.
- get_extra_sentences haalt extra zinnen op om te labellen, bijvoorbeeld woorden die nog niet vaak zijn gelabeld en zinnen die zijn gelabeld met een '4' (onzeker label).

02_analyse bevat alle modellen die ik heb gebruikt.
- BERT_train_cased_sentence_analysis.ipynb bevat het standaard BERT-model en de overige modellen en is het grootste deel van mijn experimentatie. Eerst wordt de dataset ingeladen en samengesteld. Daarna wordt de data geprept voor het model en worden er wat statistieken over de dataset gedeeld. Vervolgens wordt het model ingeladen en getraind. Er worden ook een paar analyses gedaan over de resultaten. Vervolgens wordt er een sentence analysis en een LIME-model uitgevoerd.
- GTP3.5.ipynb bevat het inladen van GPT3.5 en het testen op de dataset.
- RobBERT_train_cased.ipynb bevat de overige BERT-modellen, zoals RobBERT en sentenceBERT. Hier wordt hetzelfde mee gedaan als met de gewone BERT-modellen, behalve dat de sentence analysis en LIME hier niet op worden toegepast.


Geïnstalleerde packages:
Package                            Version
---------------------------------- -----------
absl-py                            2.1.0
accelerate                         0.20.1
aiohttp                            3.8.6
aiosignal                          1.3.1
alabaster                          0.7.12
alembic                            1.3.1
altair                             5.0.1
anaconda-client                    1.7.2
anaconda-navigator                 1.9.7
anaconda-project                   0.8.3
appdirs                            1.4.4
asn1crypto                         1.0.1
ast_decompiler                     0.7.0
astatine                           0.3.3
astor                              0.8.1
astpretty                          2.1.0
astroid                            2.15.8
astropy                            3.2.2
asttokens                          2.4.1
astunparse                         1.6.3
async-generator                    1.10
async-timeout                      4.0.3
asynctest                          0.13.0
atomicwrites                       1.3.0
attrs                              23.2.0
autoflake                          1.7.8
Babel                              2.7.0
backcall                           0.1.0
backports.cached-property          1.0.2
backports.functools-lru-cache      1.5
backports.os                       0.1.1
backports.shutil-get-terminal-size 1.0.0
backports.tempfile                 1.0
backports.weakref                  1.0.post1
backports.zoneinfo                 0.2.1
bandit                             1.7.5
beautifulsoup4                     4.8.0
biocutils                          0.0.7
bitarray                           1.0.1
bitsandbytes                       0.42.0
bkcharts                           0.2
black                              22.12.0
bleach                             3.1.0
blinker                            1.4
bokeh                              1.3.4
boto                               2.49.0
Bottleneck                         1.2.1
cachetools                         5.3.3
captum                             0.7.0
certifi                            2019.9.11
certipy                            0.1.3
cffi                               1.12.3
chardet                            3.0.4
charset-normalizer                 3.3.2
click                              8.1.7
cloudpickle                        1.2.2
clyent                             1.2.2
cognitive-complexity               1.3.0
colorama                           0.4.1
conda                              4.7.12
conda-build                        3.18.9
conda-package-handling             1.6.0
conda-verify                       3.4.2
contextlib2                        0.6.0
coverage                           6.5.0
cryptography                       2.7
cycler                             0.10.0
Cython                             0.29.13
cytoolz                            0.10.0
darglint                           1.8.1
dask                               2.5.2
datasets                           2.13.2
decorator                          4.4.0
defusedxml                         0.6.0
dill                               0.3.6
distlib                            0.3.8
distributed                        2.5.2
dlint                              0.14.1
doc8                               0.11.2
docformatter                       1.7.5
docker                             6.1.3
docker-pycreds                     0.4.0
docstring_parser                   0.16
docutils                           0.15.2
domdf-python-tools                 3.8.0.post2
entrypoints                        0.3
eradicate                          2.3.0
et-xmlfile                         1.0.1
eval-type-backport                 0.1.3
evaluate                           0.4.1
exceptiongroup                     1.2.0
fastcache                          1.1.0
fastjsonschema                     2.19.1
filelock                           3.12.2
flake8                             4.0.1
flake8-2020                        1.6.1
flake8-aaa                         0.15.0
flake8-annotations                 2.9.1
flake8-annotations-complexity      0.0.8
flake8-annotations-coverage        0.0.6
flake8-bandit                      3.0.0
flake8-black                       0.3.6
flake8-blind-except                0.2.1
flake8-breakpoint                  1.1.0
flake8-broken-line                 0.4.0
flake8-bugbear                     22.12.6
flake8-builtins                    1.5.3
flake8-class-attributes-order      0.1.3
flake8-coding                      1.3.2
flake8-cognitive-complexity        0.1.0
flake8-commas                      2.1.0
flake8-comments                    0.1.2
flake8-comprehensions              3.13.0
flake8-debugger                    4.1.2
flake8-django                      1.4
flake8-docstrings                  1.7.0
flake8-encodings                   0.5.1
flake8-eradicate                   1.4.0
flake8-executable                  2.1.3
flake8-expression-complexity       0.0.11
flake8-fixme                       1.1.1
flake8-functions                   0.0.8
flake8-functions-names             0.1.0
flake8-future-annotations          0.0.5
flake8-helper                      0.2.2
flake8-isort                       4.2.0
flake8-literal                     1.4.0
flake8-logging-format              0.9.0
flake8-markdown                    0.3.0
flake8-mutable                     1.2.0
flake8-no-pep420                   2.6.0
flake8-noqa                        1.4.0
flake8-pie                         0.16.0
flake8-plugin-utils                1.3.3
flake8-polyfill                    1.0.2
flake8-pyi                         22.11.0
flake8-pylint                      0.2.1
flake8-pytest-style                1.7.2
flake8-quotes                      3.4.0
flake8-rst-docstrings              0.2.7
flake8-secure-coding-standard      1.3.0
flake8_simplify                    0.21.0
flake8-slots                       0.1.6
flake8-string-format               0.3.0
flake8-tidy-imports                4.9.0
flake8-typing-imports              1.12.0
flake8-use-fstring                 1.4
flake8-use-pathlib                 0.3.0
flake8-useless-assert              0.4.4
flake8-variables-names             0.0.6
flake8-warnings                    0.4.0
Flask                              1.1.1
flatbuffers                        24.3.25
frozenlist                         1.3.3
fsspec                             2023.1.0
future                             0.17.1
gast                               0.4.0
gevent                             1.4.0
gitdb                              4.0.11
GitPython                          3.1.42
glob2                              0.7
gmpy2                              2.0.8
google-auth                        2.29.0
google-auth-oauthlib               0.4.6
google-pasta                       0.2.0
greenlet                           0.4.15
grpcio                             1.62.1
h5py                               2.9.0
HeapDict                           1.0.1
html5lib                           1.0.1
huggingface-hub                    0.16.4
hypothesis                         6.79.4
hypothesmith                       0.1.9
idna                               2.8
imageio                            2.6.0
imagesize                          1.1.0
importlib-metadata                 6.7.0
iniconfig                          2.0.0
ipykernel                          5.1.2
ipython                            7.8.0
ipython_genutils                   0.2.0
ipywidgets                         7.5.1
isort                              4.3.21
itsdangerous                       1.1.0
jdcal                              1.4.1
jedi                               0.15.1
jeepney                            0.4.1
Jinja2                             3.1.4
joblib                             0.13.2
json5                              0.8.5
jsonschema                         3.0.2
jupyter                            1.0.0
jupyter_client                     7.4.9
jupyter-console                    6.0.0
jupyter_core                       4.12.0
jupyterhub                         1.0.0
jupyterlab                         1.1.4
jupyterlab-flake8                  0.7.1
jupyterlab-pygments                0.2.2
jupyterlab-server                  1.0.6
keras                              2.11.0
keyring                            18.0.0
kiwisolver                         1.1.0
lark-parser                        0.12.0
lazy-object-proxy                  1.4.2
legacy                             0.1.7
libarchive-c                       2.8
libclang                           18.1.1
libcst                             0.4.10
lief                               0.9.0
lime                               0.2.0.1
llvmlite                           0.29.0
locket                             0.2.0
lxml                               4.4.1
Mako                               1.1.0
Markdown                           3.4.4
markdown-it-py                     2.2.0
MarkupSafe                         2.1.5
matplotlib                         3.1.1
mccabe                             0.6.1
mdurl                              0.1.2
mistune                            3.0.2
mkl-fft                            1.0.14
mkl-random                         1.1.0
mkl-service                        2.3.0
mock                               3.0.5
more-itertools                     7.2.0
mpmath                             1.1.0
mr-proper                          0.0.7
msgpack                            0.6.1
multidict                          6.0.5
multipledispatch                   0.6.0
multiprocess                       0.70.14
mypy-extensions                    1.0.0
natsort                            8.4.0
navigator-updater                  0.2.1
nbclient                           0.7.4
nbconvert                          7.6.0
nbformat                           5.8.0
nest-asyncio                       1.6.0
networkx                           2.3
nltk                               3.4.5
nose                               1.3.7
notebook                           6.0.1
numba                              0.45.1
numexpr                            2.7.0
numpy                              1.21.6
numpydoc                           0.9.1
nvidia-cublas-cu11                 11.10.3.66
nvidia-cuda-nvrtc-cu11             11.7.99
nvidia-cuda-runtime-cu11           11.7.99
nvidia-cudnn-cu11                  8.5.0.96
oauthlib                           3.0.1
olefile                            0.46
openpyxl                           3.0.0
opt-einsum                         3.3.0
packaging                          23.2
pamela                             1.0.0
pandas                             1.0.0
pandas-vet                         0.2.3
pandocfilters                      1.4.2
parso                              0.5.1
partd                              1.0.0
path.py                            12.0.1
pathlib2                           2.3.5
pathspec                           0.9.0
patsy                              0.5.1
pbr                                6.0.0
pdf2image                          1.16.0
peft                               0.3.0
pep517                             0.13.1
pep8                               1.7.1
pep8-naming                        0.12.1
pexpect                            4.7.0
pickleshare                        0.7.5
Pillow                             6.2.0
pip                                24.0
pkginfo                            1.5.0.1
platformdirs                       2.6.2
pluggy                             0.13.0
ply                                3.11
prefetch-generator                 1.0.3
prometheus-client                  0.7.1
prompt-toolkit                     2.0.10
protobuf                           3.19.6
psutil                             5.6.3
ptyprocess                         0.6.0
py                                 1.8.0
pyarrow                            12.0.1
pyasn1                             0.5.1
pyasn1-modules                     0.3.0
pybetter                           0.4.1
pycln                              1.3.5
pycodestyle                        2.8.0
pycosat                            0.6.3
pycparser                          2.19
pycrypto                           2.6.1
pycurl                             7.43.0.3
pydeck                             0.8.1b0
pydocstyle                         6.3.0
pyemojify                          0.2.0
pyflakes                           2.4.0
Pygments                           2.17.2
PyJWT                              1.7.1
pylint                             2.17.7
Pympler                            1.0.1
pymultihash                        0.8.2
pyodbc                             4.0.27
pyOpenSSL                          19.0.0
pyparsing                          2.4.2
pyproject                          1.3.1
pyproject-toml                     0.0.10
PyQt5                              5.12.3
PyQt5-sip                          12.13.0
PyQtWebEngine                      5.12.1
pyrsistent                         0.15.4
PySocks                            1.7.1
pytest                             7.4.4
pytest-arraydiff                   0.3
pytest-astropy                     0.5.0
pytest-cov                         3.0.0
pytest-doctestplus                 0.4.0
pytest-openfiles                   0.4.0
pytest-remotedata                  0.3.2
pytest-sugar                       0.9.7
python-dateutil                    2.9.0.post0
python-dev-tools                   2022.5.27
python-editor                      1.0.4
python-version                     0.0.2
pytz                               2019.3
pytz-deprecation-shim              0.1.0.post0
pyupgrade                          2.38.4
PyWavelets                         1.0.3
PyYAML                             6.0.1
pyzmq                              26.0.3
QtAwesome                          0.6.0
qtconsole                          4.5.5
QtPy                               1.9.0
regex                              2023.12.25
removestar                         1.5
requests                           2.31.0
requests-oauthlib                  2.0.0
responses                          0.18.0
restructuredtext-lint              1.4.0
rich                               13.7.1
rope                               0.14.0
rsa                                4.9
ruamel_yaml                        0.15.46
safetensors                        0.4.2
scikit-image                       0.15.0
scikit-learn                       0.21.3
scipy                              1.3.1
seaborn                            0.9.0
SecretStorage                      3.1.1
Send2Trash                         1.5.0
sentencepiece                      0.2.0
sentry-sdk                         1.9.0
seqeval                            1.2.2
setproctitle                       1.3.3
setuptools                         68.0.0
setuptools-scm                     7.1.0
shtab                              1.7.1
simplegeneric                      0.8.1
simpletransformers                 0.63.11
singledispatch                     3.4.0.3
six                                1.16.0
smmap                              5.0.1
snowballstemmer                    2.2.0
sortedcollections                  1.1.2
sortedcontainers                   2.1.0
soupsieve                          1.9.3
speedtest-cli                      2.1.3
Sphinx                             4.3.2
sphinxcontrib-applehelp            1.0.1
sphinxcontrib-devhelp              1.0.1
sphinxcontrib-htmlhelp             2.0.0
sphinxcontrib-jsmath               1.0.1
sphinxcontrib-qthelp               1.0.2
sphinxcontrib-serializinghtml      1.1.5
sphinxcontrib-websupport           1.1.2
spyder                             3.3.6
spyder-kernels                     0.5.2
SQLAlchemy                         1.3.9
ssort                              0.10.0
statsmodels                        0.10.1
stdlib-list                        0.10.0
stevedore                          3.5.2
streamlit                          1.23.1
sympy                              1.4
tables                             3.5.2
tblib                              1.4.0
tenacity                           8.2.3
tensorboard                        2.11.2
tensorboard-data-server            0.6.1
tensorboard-plugin-wit             1.8.1
tensorflow                         2.11.0
tensorflow-estimator               2.11.0
tensorflow-io-gcs-filesystem       0.34.0
termcolor                          2.3.0
terminado                          0.8.2
testpath                           0.4.2
tinycss2                           1.2.1
tokenize-rt                        4.2.1
tokenizers                         0.13.3
toml                               0.10.2
tomli                              2.0.1
tomlkit                            0.12.4
toolz                              0.10.0
torch                              1.13.1
tornado                            6.2
tox                                3.28.0
tox-travis                         0.13
tqdm                               4.66.2
traitlets                          5.9.0
transformers                       4.28.1
trl                                0.7.4
typed-ast                          1.4.3
typer                              0.4.2
typing_extensions                  4.7.1
typing-inspect                     0.9.0
tyro                               0.8.3
tzdata                             2024.1
tzlocal                            4.3.1
unicodecsv                         0.14.1
untokenize                         0.1.1
urllib3                            2.0.7
validators                         0.20.0
varint                             1.0.2
virtualenv                         20.16.2
wandb                              0.16.4
watchdog                           3.0.0
wcwidth                            0.1.7
webencodings                       0.5.1
websocket-client                   1.6.1
wemake-python-styleguide           0.16.1
Werkzeug                           2.2.3
wheel                              0.42.0
widgetsnbextension                 3.5.1
wrapt                              1.11.2
wurlitzer                          1.0.3
xlrd                               1.2.0
XlsxWriter                         1.2.1
xlwt                               1.3.0
xxhash                             3.4.1
yarl                               1.9.4
zict                               1.0.0
zipp                               0.6.0
