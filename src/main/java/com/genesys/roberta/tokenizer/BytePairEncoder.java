package com.genesys.roberta.tokenizer;

import lombok.NonNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Byte-Pair-Encoding
 * Relies on a pre-tokenizer that splits the training data into words, in our case space.
 *
 * This greedy algorithm looks for the best way to divide given input word.
 * It does that by dividing the word into characters, then assembles sub strings of the given word trying to find the best
 * partition of the word according to the ranks of the merges file.
 */
class BytePairEncoder {

    /**
     * Applies the byte level BPE algorithm on the given word
     *
     * @param word one word from an input sentence
     * @param robertaTokenizerRobertaResources holds the vocabulary resources
     * @return a list of strings optimally partitioned and ready for tokenization
     */
    public List<String> encode(@NonNull final String word, @NonNull RobertaTokenizerResources robertaTokenizerRobertaResources) {
        List<String> wordCharactersStrList = word.chars()
                .mapToObj(Character::toString)
                .collect(Collectors.toList());

        Set<BiGram> biGramsSet = getBiGrams(wordCharactersStrList);

        while (true) {
            long minScore = Integer.MAX_VALUE;
            BiGram lowestScoreBiGram = null;

            for (BiGram biGram : biGramsSet) {
                long score = robertaTokenizerRobertaResources.getRankOrDefault(biGram, Integer.MAX_VALUE);

                // Note that we turn the most frequent bi-gram from a max problem to minimum
                // The lower the score the higher the frequency
                if (score < minScore) {
                    minScore = score;
                    lowestScoreBiGram = biGram;
                }
            }

            // Reaching here means that only BiGrams that aren’t in the vocabulary (got rank Integer.MAX_VALUE) are left in
            // wordCharactersStrList, so no more merges should be done and the final tokenized word is the current wordCharactersStrList.
            if (lowestScoreBiGram == null) {
                break;
            }

            String first = lowestScoreBiGram.getLeft();
            String second = lowestScoreBiGram.getRight();
            List<String> newWordList = new ArrayList<>();
            int currIdx = 0;

            while (currIdx < wordCharactersStrList.size()) {
                int biGramStartIndex = getIndexWithStartPosition(wordCharactersStrList, first, currIdx);

                if (biGramStartIndex != -1) {
                    newWordList.addAll(wordCharactersStrList.subList(currIdx, biGramStartIndex));
                    currIdx = biGramStartIndex;
                } else {
                    newWordList.addAll(wordCharactersStrList.subList(currIdx, wordCharactersStrList.size()));
                    break;
                }

                if (wordCharactersStrList.get(currIdx).equals(first) && currIdx < wordCharactersStrList.size() - 1 &&
                        wordCharactersStrList.get(currIdx + 1).equals(second)) {
                    newWordList.add(first + second);
                    currIdx += 2;
                } else {
                    newWordList.add(wordCharactersStrList.get(currIdx));
                    currIdx += 1;
                }
            }

            wordCharactersStrList = newWordList;
            if (wordCharactersStrList.size() == 1) {
                break;
            } else {
                biGramsSet = getBiGrams(wordCharactersStrList);
            }
        }

        return wordCharactersStrList;
    }

    /**
     *
     * @param wordStrChars all characters of the word represented each by a String
     * @return list of all adjacent biGrams
     * e.g., "hello" will be given as input: ["h", "e", "l", "l", "o"] and will return {"he", "el","ll", "lo"}
     */
    private Set<BiGram> getBiGrams(@NonNull final List<String> wordStrChars) {
        return IntStream.range(0, wordStrChars.size() - 1)
                .mapToObj(i -> BiGram.of(wordStrChars.get(i), wordStrChars.get(i + 1)))
                .collect(Collectors.toSet());
    }

    /**
     * Looking for given word in wordCharsList and returns the index if found
     *
     * @param wordCharsList list of characters represented as Strings
     * @param word given word to search for
     * @param startPosition an index to start the search from
     * @return the index found o.w. -1
     */
    private int getIndexWithStartPosition(@NonNull final List<String> wordCharsList, @NonNull final String word,
                                          final int startPosition) {
        return IntStream.range(startPosition, wordCharsList.size())
                .filter(idx -> wordCharsList.get(idx).equals(word))
                .findFirst()
                .orElse(-1);
    }
}
