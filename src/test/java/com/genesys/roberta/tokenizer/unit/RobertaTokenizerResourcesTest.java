package com.genesys.roberta.tokenizer.unit;

import com.genesys.roberta.tokenizer.BiGram;
import com.genesys.roberta.tokenizer.RobertaTokenizer;
import com.genesys.roberta.tokenizer.RobertaTokenizerResources;
import lombok.val;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import static com.genesys.roberta.tokenizer.utils.CommonTestUtils.getResourceAbsPath;

public class RobertaTokenizerResourcesTest {

    private static final String VOCABULARY_BASE_DIR_PATH = getResourceAbsPath();

    private static final long UNKNOWN_TOKEN = RobertaTokenizer.UNK_TOKEN;
    private RobertaTokenizerResources robertaTokenizerResources;

    @BeforeClass
    public void initDataMembersBeforeClass() {
        robertaTokenizerResources = new RobertaTokenizerResources(VOCABULARY_BASE_DIR_PATH);
    }

    @Test(expectedExceptions = NullPointerException.class)
    public void nullBaseDirPath() {
        new RobertaTokenizerResources(null);
    }

    @Test(expectedExceptions = IllegalStateException.class)
    public void vocabularyBaseDirPathNotExist() {
        new RobertaTokenizerResources("dummy/base/dir/path");
    }

    @Test
    public void minByteValue() {
        byte key = -128;
        val encodedChar = robertaTokenizerResources.encodeByte(key);
        Assert.assertNotNull(encodedChar);
    }

    @Test
    public void maxByteValue() {
        byte key = 127;
        val encodedChar = robertaTokenizerResources.encodeByte(key);
        Assert.assertNotNull(encodedChar);
    }

    @Test
    public void wordDoesNotExist() {
        String word = "Funnel";
        Long actualToken = robertaTokenizerResources.encodeWord(word, UNKNOWN_TOKEN);
        Assert.assertEquals(actualToken.longValue(), UNKNOWN_TOKEN);
    }

    @Test
    public void wordExists() {
        String word = "er";
        long expectedToken = 19;
        Long actualToken = robertaTokenizerResources.encodeWord(word, UNKNOWN_TOKEN);
        Assert.assertEquals(actualToken.longValue(), expectedToken);
    }

    @Test
    public void pairExists() {
        BiGram bigram = BiGram.of("e", "r");
        int actualRank = robertaTokenizerResources.getRankOrDefault(bigram, Integer.MAX_VALUE);
        int expectedRank = 3;
        Assert.assertEquals(actualRank, expectedRank);
    }

    @Test
    public void pairDoesNotExist() {
        BiGram bigram = BiGram.of("Zilpa", "Funnel");
        int actualRank = robertaTokenizerResources.getRankOrDefault(bigram, Integer.MAX_VALUE);
        Assert.assertEquals(actualRank, Integer.MAX_VALUE);
    }
}
