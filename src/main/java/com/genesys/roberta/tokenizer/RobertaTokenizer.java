package com.genesys.roberta.tokenizer;

import lombok.NonNull;
import lombok.val;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.LongStream;

import static java.util.stream.LongStream.concat;
import static java.util.stream.LongStream.of;

/**
 * Tokenizer used for the RoBERTa model.
 * Encode sentences to integer tokens.
 *
 * This tokenizer is implemented according to the following:
 * - https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer
 * - https://github.com/hyunwoongko/gpt2-tokenizer-java/blob/master/src/main/java/ai/tunib/tokenizer/GPT2Tokenizer.java
 * - https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2/src/main/java/co/huggingface/android_transformers/\
 *   gpt2/tokenization/GPT2Tokenizer.kt
 */
public class RobertaTokenizer implements Tokenizer {

    public static final long DEFAULT_CLS_TOKEN = 0;
    public static final long DEFAULT_SEP_TOKEN = 2;
    public static final long DEFAULT_UNK_TOKEN = 3;

    //splits a given sentence by space in to words or sub-words
    private static final Pattern PATTERN = Pattern
            .compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    // Special tokens
    private final long clsToken; // Also BOS (beginning of sequence) token
    private final long sepToken; // Also EOS (end of sequence) token
    private final long unkToken; // Unknown Token.

    private final RobertaTokenizerResources robertaResources;
    private final BytePairEncoder bytePairEncoder;

    /**
     * Constructs a RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.
     *
     * @param robertaTokenizerResources - responsible for providing roberta vocabularies and merges files.
     *
     * Note that this constructor will use HuggingFace's default special tokens:
     * [CLS_TOKEN = 0, SEP_TOKEN = 2, UNK_TOKEN = 3]
     */
    public RobertaTokenizer(@NonNull final RobertaTokenizerResources robertaTokenizerResources) {
        this(robertaTokenizerResources, DEFAULT_CLS_TOKEN, DEFAULT_SEP_TOKEN, DEFAULT_UNK_TOKEN);
    }

    /**
     * Constructs a RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.
     *
     * @param robertaTokenizerResources - responsible for providing roberta vocabularies and merges files.
     * @param clsToken Classification token
     * @param sepToken Separator token
     * @param unkToken Unknown token
     */
    public RobertaTokenizer(@NonNull final RobertaTokenizerResources robertaTokenizerResources, final long clsToken,
                            final long sepToken, final long unkToken) {
        this.robertaResources = robertaTokenizerResources;
        this.bytePairEncoder = new BytePairEncoder();
        this.clsToken = clsToken;
        this.sepToken = sepToken;
        this.unkToken = unkToken;
    }

    /**
     * Encodes the given word into a list of tokens (long numbers) using Byte Level Byte-Pair-Encoding.
     *
     * @param sentence a word or more divided by space
     * @return an array of tokens (long) values
     */
    @Override
    public long[] tokenize(@NonNull final String sentence) {
        List<String> encodedStrings = new ArrayList<>();

        Matcher matcher = PATTERN.matcher(sentence);
        while (matcher.find()) {
            String matchedSequence = matcher.group();
            val matchedSequenceEncoded = new StringBuilder();

            for (byte b : matchedSequence.getBytes(StandardCharsets.UTF_8)) {
                String encodedByte = this.robertaResources.encodeByte(b);
                matchedSequenceEncoded.append(encodedByte);
            }

            encodedStrings.add(matchedSequenceEncoded.toString());
        }

        LongStream outputTokens = encodedStrings.stream()
                // returns list of strings ready for vocabulary mapping
                .map(encodedStr -> bytePairEncoder.encode(encodedStr, robertaResources))
                // mapping each word in the given lists to a Long token from the vocabulary
                .flatMapToLong(encodedStrList -> encodedStrList.stream()
                        .mapToLong(word -> this.robertaResources.encodeWord(word, unkToken)));

        outputTokens = concat(of(clsToken), outputTokens); // adding BOS
        return concat(outputTokens, of(sepToken)).toArray(); // adding EOS
    }

    public long getClsToken() {
        return clsToken;
    }

    public long getSepToken() {
        return sepToken;
    }

    public long getUnkToken() {
        return unkToken;
    }
}
