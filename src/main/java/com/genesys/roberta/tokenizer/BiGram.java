package com.genesys.roberta.tokenizer;

import lombok.NonNull;
import org.apache.commons.lang3.tuple.Pair;

import static com.google.common.base.Preconditions.checkState;
import static java.lang.String.format;

public class BiGram extends Pair<String, String> {
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

    @Override
    public String getLeft() {
        return this.left;
    }

    @Override
    public String getRight() {
        return this.right;
    }

    @Override
    public String setValue(@NonNull final String value) {
        throw new UnsupportedOperationException();
    }
}
