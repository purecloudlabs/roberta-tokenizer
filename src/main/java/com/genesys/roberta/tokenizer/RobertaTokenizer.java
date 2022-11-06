package com.genesys.roberta.tokenizer;

import lombok.NonNull;
import lombok.val;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
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

    // Tokenized sentences shorter than the max allowed length will be padded with the PAD_TOKEN
    public static final long PAD_TOKEN = 1;

    // Default of Hugging Face
    public static final long CLS_TOKEN = 0; // also BOS - beginning of sequence
    public static final long SEP_TOKEN = 2; // also EOS - end of sequence
    public static final long UNK_TOKEN = 3;
    //splits a given sentence by space in to words or sub-words
    private static final Pattern PATTERN = Pattern
            .compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    private final RobertaTokenizerResourcesFactory robertaTokenizerFactory;
    private final AtomicReference<RobertaTokenizerResources> robertaResourcesCache = new AtomicReference<>();
    private final BytePairEncoder bytePairEncoder;

    /**
     * Constructs a RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.
     *
     * @param robertaTokenizerResourcesFactory - responsible for providing roberta vocabularies and merges files.
     */
    public RobertaTokenizer(@NonNull final RobertaTokenizerResourcesFactory robertaTokenizerResourcesFactory) {
        this.robertaTokenizerFactory = robertaTokenizerResourcesFactory;
        this.bytePairEncoder = new BytePairEncoder();
    }

    @Override
    public long[] tokenize(@NonNull final String sentence) {
        RobertaTokenizerResources robertaResources = getRobertaTokenizerResources();
        return encode(sentence, robertaResources);
    }

    /**
     * This method provides the resources needed for tokenization.
     * It's initialized only when used first time
     *
     * @return object which holds the vocabularies and merges file according to the existing baseDirPath
     */
    private RobertaTokenizerResources getRobertaTokenizerResources() {
        if (robertaResourcesCache.get() == null) {
            robertaResourcesCache.compareAndSet(null, robertaTokenizerFactory.create());
        }
        return robertaResourcesCache.get();
    }

    /**
     * Encodes the given word into a list of tokens (long numbers) using Byte Level Byte-Pair-Encoding.
     *
     * @param text a sentence
     * @return an array of tokens (long) values
     */
    private long[] encode(@NonNull final String text, @NonNull final RobertaTokenizerResources robertaResources) {
        List<String> encodedStrings = new ArrayList<>();

        Matcher matcher = PATTERN.matcher(text);
        while (matcher.find()) {
            String matchedSequence = matcher.group();
            val matchedSequenceEncoded = new StringBuilder();

            for (byte b : matchedSequence.getBytes(StandardCharsets.UTF_8)) {
                String encodedByte = robertaResourcesCache.get().encodeByte(b);
                matchedSequenceEncoded.append(encodedByte);
            }

            encodedStrings.add(matchedSequenceEncoded.toString());
        }

        LongStream outputTokens = encodedStrings.stream()
                // returns list of strings ready for vocabulary mapping
                .map(encodedStr -> bytePairEncoder.encode(encodedStr, robertaResources))
                // mapping each word in the given lists to a Long token from the vocabulary
                .flatMapToLong(encodedStrList -> encodedStrList.stream()
                        .mapToLong(word -> robertaResourcesCache.get().encodeWord(word, UNK_TOKEN)));

        return concat(
                concat(of(CLS_TOKEN), outputTokens), // adding BOS
                of(SEP_TOKEN)) // adding EOS
                .toArray();
    }
}
