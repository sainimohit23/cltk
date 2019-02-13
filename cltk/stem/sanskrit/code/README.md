# Sanskrit Sandhi Splitter
This directory contains modified code and data for the following paper:

Oliver Hellwig, Sebastian Nehrdich: Sanskrit Word Segmentation Using Character-level Recurrent and Convolutional Neural Networks. In: Proceedings of the EMNLP 2018.

## Dependencies
* tensorflow
* numpy
* os
* git
* h5py
* json

## Working
It takes sanskrit text as an input and returns a dictionary as output. Each word of input( ex-`पुद्गलधर्मनैरात्म्ययोर्`) sanskrit text act as a key for the dictionary and value of each key is list of sandhi splitted words (`['पुद्गल', 'धर्म', 'नैरात्म्ययोः']`) for the corresponding key. On abstract level program has three main steps:

**1)** Encoding of input sanskrit text to it's corresponding [IAST](https://en.wikipedia.org/wiki/Devanagari_transliteration#IAST) encoding. It is done using the script `dev_iast.py`.

original sanskrit text:
```
पुद्गलधर्मनैरात्म्ययोर्
अप्रतिपन्नविप्रतिपन्नानाम्
अविपरीतपुद्गलधर्मनैरात्म्यप्रतिपादनार्थं
त्रिंशिकाविज्ञप्तिप्रकरणारम्भः 
पुद्गलधर्मनैरात्म्यप्रतिपादनं 
पुनः 
क्लेशज्ञेयावरणप्रहाणार्थम्

```

Sanskrit text encoded in IAST:
```
pudgaladharmanairātmyayor
apratipannavipratipannānām
aviparītapudgaladharmanairātmyapratipādanārthaṃ
triṃśikāvijñaptiprakaraṇārambhaḥ  
pudgaladharmanairātmyapratipādanaṃ
punaḥ
kleśajñeyāvaraṇaprahāṇārtham 

```




**2)** Now this encoded text is passed through a pre trained character level tensorflow model which splits the sandhi of each input word. The output of the ML model when above IAST encoded text is used as input:

```
pudgala-dharma-nairātmyayoḥ
apratipanna-vipratipannānām
aviparīta-pudgala-dharma-nairātmya-pratipādana-artham
triṃśikā-vijñapti-prakaraṇa-ārambhaḥ
pudgala-dharma-nairātmya-pratipādanam
punar 
kleśa-jñeya-āvaraṇa-prahāṇa-artham

```



**3)** In last step we use `iast_dv.py` to decode the output of the model from IAST to devnagiri.
```
पुद्गल-धर्म-नैरात्म्ययोः
अप्रतिपन्न-विप्रतिपन्नानाम्
अविपरीत-पुद्गल-धर्म-नैरात्म्य-रतिपादन-अर्थम्
त्रिंशिका-विज्ञप्ति-प्रकरण-आरम्भः
पुद्गल-धर्म-नैरात्म्ययोः
पुनर्
क्लेश-ज्ञेय-आवरण-प्रहाण-अर्थम्

```

**Last Step** : In last step some post processing is done to return above output as a dictionary.
```
{'पुनः': ['पुनर्'], 
'पुद्गलधर्मनैरात्म्ययोर्': ['पुद्गल', 'धर्म', 'नैरात्म्ययोः'],
'क्लेशज्ञेयावरणप्रहाणार्थम्': ['क्लेश', 'ज्ञेय', 'आवरण', 'प्रहाण', 'अर्थम्'],
'अविपरीतपुद्गलधर्मनैरात्म्यप्रतिपादनार्थं': ['अविपरीत', 'पुद्गल', 'धर्म', 'नैरात्म्य', 'प्रतिपादन', 'अर्थम्'], 
'अप्रतिपन्नविप्रतिपन्नानाम्': ['अप्रतिपन्न', 'विप्रतिपन्नानाम्'], 
'त्रिंशिकाविज्ञप्तिप्रकरणारम्भः': ['त्रिंशिका', 'विज्ञप्ति', 'प्रकरण', 'आरम्भः'], 
'पुद्गलधर्मनैरात्म्यप्रतिपादनं': ['पुद्गल', 'धर्म', 'नैरात्म्य', 'प्रतिपादनम्']}
```

## Usage
`sandhisplitter.py` contains the class `SandhiSplitter` which can perform above three steps.It can be initialized in two ways.
1) When input text is in devnagiri script. ex- `splitter=SandhiSplitter(text='अविपरीतपुद्गलधर्मनैरात्म्यप्रतिपादनार्थं',isIAST=False)`
 
    default value of isIAST = False
 
 2) When input is IAST encoded. ex- `splitter=SandhiSplitter(text='aviparītapudgaladharmanairātmyapratipādanārthaṃ', isIAST=True)`

* Now to get sandhi dictionary we just have to call the `getSandhi()` method. ex-

  `print(splitter.getSandhi())`
 
  **output:** `{'पुद्गलधर्मनैरात्म्ययोर्': ['पुद्गल', 'धर्म', 'नैरात्म्ययोः']}`
  
  **NOTE**: If model is not found locally then getSandhi() method will download the pre trained model first.
