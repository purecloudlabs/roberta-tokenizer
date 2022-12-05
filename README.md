# RoBERTa Java Tokenizer #


## About

---
This repo contains a Java tokenizer used by RoBERTa model. The implementation is mainly according to HuggingFace Python
RoBERTa Tokenizer, but also we took references from other implementations as mentioned in the code and below:

* https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer

* https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2/src/main/java/co/huggingface/android_transformers/gpt2/tokenization/GPT2Tokenizer.kt

* https://github.com/hyunwoongko/gpt2-tokenizer-java/blob/master/src/main/java/ai/tunib/tokenizer/GPT2Tokenizer.java

The algorithm used is a byte-level Byte Pair Encoding.

https://huggingface.co/docs/transformers/tokenizer_summary#bytelevel-bpe
## How do I get set up? ###

---

* Clone the repo for explicit usage.
* Add the Maven dependency to your `pom.xml` for usage in your project:

```
<dependency>
    <groupId>cloud.genesys</groupId>
    <artifactId>roberta-tokenizer</artifactId>
    <version>1.0.5</version>
</dependency>

<distributionManagement>
    <repository>
      <id>ossrh</id>
      <url>https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/</url>
    </repository>
    ...
</distributionManagement>
```


### Tests ###

---

* Unit tests - Run on local machine.

### File Dependencies ###

---

Since we want efficiency when initializing the tokenizer, we use a factory to create the relevant resources
files and create it "lazily".

For this tokenizer we need 3 data files:

* `base_vocabulary.json` -  map of numbers ([0,255]) to symbols (UniCode Characters). Only those symbols will be known by the
  algorithm. e.g., given _s_ as input it iterates over the bytes of the String _s_ and replaces each given byte with the mapped symbol.
  This way we assure what characters are passed.

* `vocabulary.json` - Is a file that holds all the words(sub-words) and their token according to training.

* `merges.txt` - describes the merge rules of words. The algorithm splits the given word into two subwords, afterwards
  it decides the best split according to the rank of the sub words. The higher those words are, the higher the rank.

__Please note__:

1. All three files must be under the same directory.

2. They must be named like mentioned above.

3. The result of the tokenization depends on the vocabulary and merges files.

### Example ###

---

```

String baseDirPath = "base/dir/path";
RobertaTokenizerResources robertaResources = new RobertaTokenizerResources(baseDirPath);
RobertaTokenizer robertaTokenizer = new RobertaTokenizer(robertaResources);
...
String sentence = "this must be the place";
long[] tokenizedSentence = robertaTokenizer.tokenize(sentence);
System.out.println(tokenizedSentence);

```

An example output would be: `[0, 9226, 531, 28, 5, 317, 2]` - Depends on the given vocabulary and merges files.

### Contribution guidelines

---

* Use temporary branches for every issue/task.
