package com.genesys.roberta.tokenizer;

import lombok.EqualsAndHashCode;
import lombok.NonNull;

import static com.google.common.base.Preconditions.checkState;
import static java.lang.String.format;

/**
 * A sequence of two adjacent elements from a string which differs by their position - left or right
 */
@EqualsAndHashCode
class BiGram {
    private static final int PAIR_SIZE = 2;
    private final String left;
    private final String right;

    private BiGram(@NonNull final String[] pairArray) {
        checkState(pairArray.length == PAIR_SIZE,
                format("Expecting given input pair to be exactly of size [%d], but it's of size: [%d]", PAIR_SIZE, pairArray.length));
        this.left = pairArray[0];
        this.right = pairArray[1];
    }

    private BiGram(@NonNull final String left, @NonNull final String right) {
        this.left = left;
        this.right = right;
    }

    /**
     * Creates a new BiGram object from an array of Strings.
     * Expecting the array to be of size 2.
     * @param pairArray array of Strings
     * @return new BiGram where the String pairArray[0] will be left and pairArray[1] right.
     */
    public static BiGram of(@NonNull final String[] pairArray) {
        return new BiGram(pairArray);
    }

    /**
     * Creates an object with given parameters.
     * @param left will be the left String of the BiGRam
     * @param right will be the right String of the BiGRam
     * @return new BiGram object
     */
    public static BiGram of(@NonNull final String left, @NonNull final String right) {
        return new BiGram(left, right);
    }

    public String getLeft() {
        return this.left;
    }

    public String getRight() {
        return this.right;
    }
}
