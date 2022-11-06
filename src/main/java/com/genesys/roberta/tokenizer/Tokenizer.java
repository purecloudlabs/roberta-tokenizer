package com.genesys.roberta.tokenizer;

public interface Tokenizer {

    /**
     * Converts given input sentence to an array of long tokens
     *
     * @param sentence One or more words delimited by space
     * @return list of input IDs with the appropriate tokens
     */
    long[] tokenize(String sentence);
}
