package com.genesys.roberta.tokenizer;

/**
 * The use of this interface will have the ability to tokenize given String inputs
 */
interface Tokenizer {

    /**
     * Converts given input sentence to an array of long tokens
     *
     * @param sentence One or more words delimited by space
     * @return list of input IDs with the appropriate tokens
     */
    long[] tokenize(String sentence);
}
