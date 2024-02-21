package com.genesys.roberta.tokenizer;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.NonNull;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.google.common.base.Preconditions.checkState;

/**
 * Holds the vocabularies and the merges file used to encode and tokenize the inputs.
 */
public class RobertaTokenizerResources {

    private static final String BASE_VOCABULARY_FILE_NAME = "base_vocabulary.json";
    private static final String VOCABULARY_FILE_NAME = "vocabulary.json";
    private static final String MERGES_FILE_NAME = "merges.txt";

    private final Map<Integer, String> baseVocabularyMap;
    private final Map<String, Long> vocabularyMap;
    private final Map<BiGram, Integer> bpeRanks;

    /**
     * @param resourcesPath expecting this path to hold (with their names):
     *  Base Vocabulary - base_vocabulary.txt
     *  Vocabulary - vocabulary.json
     *  Merges - merges.txt
     */
    public RobertaTokenizerResources(@NonNull final String resourcesPath) {
        this.baseVocabularyMap = loadBaseVocabulary(resourcesPath);
        this.vocabularyMap = loadVocabulary(resourcesPath);
        this.bpeRanks = loadMergesFile(resourcesPath);
    }

    private Map<Integer, String> loadBaseVocabulary(@NonNull final String resourcesPath) {
        final Path baseVocabPath = Paths.get(resourcesPath, BASE_VOCABULARY_FILE_NAME);
        try {
            checkPathExists(baseVocabPath,
                    String.format("base vocabulary file path for Roberta: [ %s ] was not found", baseVocabPath));
            final Map<Integer, String> baseVocabMap = new Gson()
                    .fromJson(Files.readString(baseVocabPath), new TypeToken<HashMap<Integer, String>>(){}.getType());
            return Collections.unmodifiableMap(baseVocabMap);
        } catch (IOException e) {
            throw new IllegalStateException(String.format(
                    "Failed to load base vocabulary map for Roberta from [ %s ]", baseVocabPath), e);
        }
    }

    private Map<String, Long> loadVocabulary(@NonNull final String resourcesPath) {
        final Path vocabPath = Paths.get(resourcesPath, VOCABULARY_FILE_NAME);
        try {
            checkPathExists(vocabPath,
                    String.format("vocabulary file path for Roberta: [%s] was not found", vocabPath));
            final Map<String, Long> vocabMap = new Gson()
                    .fromJson(Files.readString(vocabPath), new TypeToken<HashMap<String, Long>>(){}.getType());
            return Collections.unmodifiableMap(vocabMap);
        } catch (IOException e) {
            throw new IllegalStateException(String.format(
                    "Failed to load vocabulary for Roberta from file path [ %s ]", vocabPath), e);
        }
    }

    /**
     * This method allows merges file to be with or without the header.
     * Other than that, it will accept in every line one BiGram ONLY, split by one space.
     *
     * @param resourcesPath resources dir path
     * @return the merges map
     */
    private Map<BiGram, Integer> loadMergesFile(@NonNull final String resourcesPath) {
        final Path mergesPath = Paths.get(resourcesPath, MERGES_FILE_NAME);
        try {
            checkPathExists(mergesPath,
                    String.format("%s merges file path: [%s] was not found", RobertaTokenizerResources.class.getSimpleName(),
                            mergesPath));

            final List<String> lines = Files.readAllLines(mergesPath, StandardCharsets.UTF_8);
            final int startIndex = isMergesFileWithHeader(lines) ? 1 : 0;

            return IntStream.range(startIndex, lines.size()).boxed()
                    .collect(Collectors.toUnmodifiableMap(idx -> BiGram.of(lines.get(idx).split(" ")), Function.identity()));
        } catch (IOException e) {
            throw new IllegalStateException(String.format(
                    "Failed to load merges file for Roberta from file path [ %s ]", mergesPath), e);
        }
    }

    /**
     * Encoding the given key to a mapped String which represents a character from the base vocabulary.
     * Since the input is of type byte values we except only values [-127, 128].
     * Shifting the range with the unsigned int operation to [0, 255]  \
     * the exact size of our base vocab map - what assures us valid input.
     *
     * @param key - byte to encode
     * @return associated String according to the base vocabulary json
     */
    public String encodeByte(final byte key) {
        // In case the byte is negative we add to it 256 by a Bitwise AND so it will be in range [0, 255]
        // This solution was taken from the below StackOverflow thread
        // https://stackoverflow.com/questions/22575308/getbytes-returns-negative-number/22575346#22575346
        return baseVocabularyMap.get(Byte.toUnsignedInt(key));
    }

    /**
     * Converts a word into an integer (long) according to the word vocabulary file
     * @param word (or subword) after bpe was applied on it
     * @param defaultValue positive integer
     * @return mapped token according to the vocabulary or default value  if it didn't exist
     */
    public Long encodeWord(@NonNull final String word, final long defaultValue) {
        return vocabularyMap.getOrDefault(word, defaultValue);
    }

    /**
     * Returns the rank for the given BiGram according to the rank file
     * @param biGram a pair of Strings
     * @param defaultValue positive integer
     * @return the rank of that pair or default value if it doesn't exist
     */
    public Integer getRankOrDefault(@NonNull final BiGram biGram, final int defaultValue) {
        return bpeRanks.getOrDefault(biGram, defaultValue);
    }

    /**
     * Since we use HuggingFace tokenizers, the merges file output might have a comment in the head of the file like:
     * "#version: 0.2 - Trained by `huggingface/tokenizers`"
     *
     * @param lines - all lines of the merges file
     * @return true if merges file starts with a comment and false o.w.
     */
    private boolean isMergesFileWithHeader(@NonNull final List<String> lines) {
        checkState(!lines.isEmpty(), "provided empty merges file");
        final String header = lines.get(0);
        return header.split(" ").length != 2;
    }

    private static void checkPathExists(final Path path, final String errorMsg) throws FileNotFoundException {
        if (!Files.exists(path)) {
            throw new FileNotFoundException(errorMsg);
        }
    }
}
