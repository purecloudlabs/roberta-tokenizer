package com.genesys.roberta.tokenizer.unit;

import com.genesys.roberta.tokenizer.logic.BiGram;
import com.genesys.roberta.tokenizer.logic.BytePairEncoder;
import com.genesys.roberta.tokenizer.resources.RobertaTokenizerResources;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.when;


public class BytePairEncoderTest {

    private BytePairEncoder bytePairEncoder;
    private Map<BiGram, Integer> ranks;

    @Mock
    private RobertaTokenizerResources robertaTokenizerResources;

    @BeforeClass
    public void setupBeforeClass() {
        MockitoAnnotations.openMocks(this);
        ranks = new HashMap<>() {{
            put(BiGram.of("Ġ", "l"), 0);
            put(BiGram.of("Ġl", "o"), 1);
            put(BiGram.of("Ġlo", "w"), 2);
            put(BiGram.of("e", "r"), 3);
        }};
        bytePairEncoder = new BytePairEncoder();
        when(robertaTokenizerResources.getRankOrDefault(any(BiGram.class), anyInt()))
                .thenAnswer(input -> ranks.getOrDefault(input.getArgument(0), Integer.MAX_VALUE));
    }

    @Test(expectedExceptions = NullPointerException.class)
    public void nullWordTest() {
        bytePairEncoder.encode(null, robertaTokenizerResources);
    }

    @Test
    public void correctSplitTest() {
        List<String> actualSplit = bytePairEncoder.encode("lowerĠnewer", robertaTokenizerResources);
        // The vocabulary rules and characters were taken from here:
        // https://github.com/huggingface/transformers/blob/v4.20.1/tests/models/roberta/test_tokenization_roberta.py#L86
        List<String> expectedSplit = Arrays.asList("l", "o", "w", "er", "Ġ", "n", "e", "w", "er");
        Assert.assertEquals(actualSplit, expectedSplit);
    }

    @Test
    public void emptySplitTest() {
        List<String> actualSplit = bytePairEncoder.encode("", robertaTokenizerResources);
        Assert.assertTrue(actualSplit.isEmpty());
    }

    @Test
    public void noMergeRulesForWordTest() {
        List<String> actualSplit = bytePairEncoder.encode("qpyt", robertaTokenizerResources);
        // Since all these characters do not appear at all at the ranks map i.e., no merge rule for any of them
        // we would expect each one to be encoded alone
        List<String> expectedSplit = Arrays.asList("q", "p", "y", "t");
        Assert.assertEquals(actualSplit, expectedSplit);
    }
}
