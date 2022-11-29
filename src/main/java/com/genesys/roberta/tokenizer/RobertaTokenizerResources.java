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
        Path baseVocabPath = Paths.get(resourcesPath, BASE_VOCABULARY_FILE_NAME);
        try {
            checkPathExists(baseVocabPath,
                    String.format("base vocabulary file path for Roberta: [ %s ] was not found", baseVocabPath));
            Map<Integer, String> baseVocabMap = new Gson()
                    .fromJson(Files.readString(baseVocabPath), new TypeToken<HashMap<Integer, String>>(){}.getType());
            return Collections.unmodifiableMap(baseVocabMap);
        } catch (IOException e) {
            throw new IllegalStateException(String.format(
                    "Failed to load base vocabulary map for Roberta from [ %s ]", baseVocabPath), e);
        }
    }

    private Map<String, Long> loadVocabulary(@NonNull final String resourcesPath) {
        Path vocabPath = Paths.get(resourcesPath, VOCABULARY_FILE_NAME);
        try {
            checkPathExists(vocabPath,
                    String.format("vocabulary file path for Roberta: [%s] was not found", vocabPath));
            Map<String, Long> vocabMap = new Gson()
                    .fromJson(Files.readString(vocabPath), new TypeToken<HashMap<String, Long>>(){}.getType());
            return Collections.unmodifiableMap(vocabMap);
        } catch (IOException e) {
            throw new IllegalStateException(String.format(
                    "Failed to load vocabulary for Roberta from file path [ %s ]", vocabPath), e);
        }
    }

    private Map<BiGram, Integer> loadMergesFile(@NonNull final String resourcesPath) {
        Path mergesPath = Paths.get(resourcesPath, MERGES_FILE_NAME);
        try {
            checkPathExists(mergesPath,
                    String.format("%s merges file path: [%s] was not found", RobertaTokenizerResources.class.getSimpleName(),
                            mergesPath));

            List<String> lines = Files.readAllLines(mergesPath, StandardCharsets.UTF_8);
            return IntStream.range(0, lines.size())
                    .boxed()
                    .collect(Collectors.toUnmodifiableMap(
                            idx -> BiGram.of(lines.get(idx).split(" ")),
                            Function.identity()));
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
     *  Returns the ranl for the given BiGram according to the rank file
     * @param biGram a pair of Strings
     * @param defaultValue positive integer
     * @return the rank of that pair or default value if it doesn't exist
     */
    public Integer getRankOrDefault(@NonNull final BiGram biGram, final int defaultValue) {
        return bpeRanks.getOrDefault(biGram, defaultValue);
    }

    private static void checkPathExists(Path path, String errorMsg) throws FileNotFoundException {
        if (!Files.exists(path)) {
            throw new FileNotFoundException(errorMsg);
        }
    }
}
