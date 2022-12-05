package com.genesys.roberta.tokenizer;

import lombok.NonNull;
import lombok.val;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.LongStream;

import static com.google.common.base.Preconditions.checkState;
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

    static long CLS_TOKEN; // Classification token. Also BOS (beginning of sequence) token
    static long SEP_TOKEN; // Separator token. Also EOS (end of sequence) token
    static long UNK_TOKEN; // Unknown Token.
    private static final int SPECIAL_TOKENS_SIZE = 3; // Unknown Token.

    //splits a given sentence by space in to words or sub-words
    private static final Pattern PATTERN = Pattern
            .compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

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
        this.robertaResources = robertaTokenizerResources;
        this.bytePairEncoder = new BytePairEncoder();
        initSpecialTokens();
    }

    /**
     * Constructs a RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.
     *
     * @param robertaTokenizerResources - responsible for providing roberta vocabularies and merges files.
     * @param specialTokens - MUST BE OF SIZE 4 and in this order: [CLS_TOKEN, SEP_TOKEN, UNK_TOKEN]
     */
    public RobertaTokenizer(@NonNull final RobertaTokenizerResources robertaTokenizerResources,
                            @NonNull final long[] specialTokens) {
        this.robertaResources = robertaTokenizerResources;
        this.bytePairEncoder = new BytePairEncoder();
        checkState(specialTokens.length == SPECIAL_TOKENS_SIZE,
                String.format("Expecting %d special tokens but received %d", SPECIAL_TOKENS_SIZE, specialTokens.length));
        initSpecialTokens(specialTokens);
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
                        .mapToLong(word -> this.robertaResources.encodeWord(word, UNK_TOKEN)));

        outputTokens = concat(of(CLS_TOKEN), outputTokens); // adding BOS
        return concat(outputTokens, of(SEP_TOKEN)).toArray(); // adding EOS
    }

    @SuppressWarnings("checkstyle:MagicNumber")
    private void initSpecialTokens() {
        initSpecialTokens(new long[]{0, 2, 3});
    }

    private void initSpecialTokens(long[] specialTokens) {
        CLS_TOKEN = specialTokens[0];
        SEP_TOKEN = specialTokens[1];
        UNK_TOKEN = specialTokens[2];
    }
}
