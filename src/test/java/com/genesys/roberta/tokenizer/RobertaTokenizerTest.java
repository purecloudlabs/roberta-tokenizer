package com.genesys.roberta.tokenizer;

import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.Arrays;

import static com.genesys.roberta.tokenizer.utils.CommonTestUtils.getResourceAbsPath;

public class RobertaTokenizerTest {
    private static final String VOCABULARY_BASE_DIR_PATH = getResourceAbsPath();
    private long clsToken;
    private long sepToken;

    @Mock
    private RobertaTokenizer robertaTokenizer;

    @BeforeClass
    public void initDataMembersBeforeClass() {
        MockitoAnnotations.openMocks(this);
        RobertaTokenizerResources robertaResources = new RobertaTokenizerResources(VOCABULARY_BASE_DIR_PATH);
        robertaTokenizer = new RobertaTokenizer(robertaResources);
        clsToken = robertaTokenizer.getClsToken();
        sepToken = robertaTokenizer.getSepToken();
    }

    @Test(expectedExceptions = NullPointerException.class)
    public void nullResourcesFactory() {
        new RobertaTokenizer(null);
    }

    @Test
    public void longSentenceWithTruncating() {
        // er token is 19, this sentence holds 24 occurrences of "er"
        String sentence = "erererererererererererererererererererererererer";
        long expectedToken = 19;
        long[] actualEncoding = robertaTokenizer.tokenize(sentence);
        Assert.assertEquals(actualEncoding[0], clsToken);
        Assert.assertTrue(Arrays.stream(actualEncoding).skip(1).takeWhile(token -> token != sepToken)
                .allMatch(token -> token == expectedToken));
        Assert.assertEquals(actualEncoding[actualEncoding.length - 1], sepToken);
    }

    @Test
    public void addingBeginningAndEndTokensToSentence() {
        String sentence = "er";
        long expectedToken = 19;
        long[] actualTokens = robertaTokenizer.tokenize(sentence);
        Assert.assertEquals(actualTokens[0], clsToken);
        Assert.assertEquals(actualTokens[1], expectedToken);
        Assert.assertEquals(actualTokens[2], sepToken);
    }

    /**
     * Since this sentence is well-defined according to the vocabulary, we know what tokens to expect.
     * Taken from here:
     * https://github.com/huggingface/transformers/blob/v4.20.1/tests/models/roberta/test_tokenization_roberta.py#L94
     */
    @Test
    public void tokenizeCorrectly() {
        String sentence = "lower newer";
        long[] expectedTokens = {
                clsToken,
                4, 5, 6, 19, // lower
                114, 13, 7, 6, 19, // newer
                sepToken};
        long[] actualTokens = robertaTokenizer.tokenize(sentence);
        Assert.assertEquals(actualTokens, expectedTokens);
    }

    @Test
    public void emptySentence() {
        long[] actualTokens = robertaTokenizer.tokenize("");
        Assert.assertEquals(actualTokens[0], clsToken);
        Assert.assertEquals(actualTokens[1], sepToken);
    }

    @Test
    public void veryLongWord() {
        String originalText =
                "https://www.google.com/search?as_q=you+have+to+write+a+really+really+long+search+to+get+to+2000+" +
                        "characters.+like+seriously%2C+you+have+no+idea+how+long+it+has+to+be&as_epq=2000+characters+" +
                        "is+absolutely+freaking+enormous.+You+can+fit+sooooooooooooooooooooooooooooooooo+much+data+" +
                        "into+2000+characters.+My+hands+are+getting+tired+typing+this+many+characters.+I+didn%27t+" +
                        "even+realise+how+long+it+was+going+to+take+to+type+them+all.&as_oq=Argh!+So+many+" +
                        "characters.+I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.+I%27m+bored+now%2C+so+I%27ll+" +
                        "just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+" +
                        "so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+" +
                        "now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+" +
                        "bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+" +
                        "paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+" +
                        "copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+" +
                        "I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+" +
                        "now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+" +
                        "bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+" +
                        "paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+" +
                        "copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+" +
                        "I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+" +
                        "now%2C+so+I%27ll+just+copy+and+paste.I%27m+bored+now%2C+so+I%27ll+just+copy+and+" +
                        "paste.&as_eq=It+has+to+be+freaking+enormously+freaking+enormous&as_nlo=123&as_nhi=456&lr=" +
                        "lang_hu&cr=countryAD&as_qdr=m&as_sitesearch=stackoverflow.com&as_occt=title&safe=active&tbs=" +
                        "rl%3A1%2Crls%3A0&as_filetype=xls&as_rights=(cc_publicdomain%7Ccc_attribute%7Ccc_sharealike%" +
                        "7Ccc_nonderived).-(cc_noncommercial)&gws_rd=ssl";
        robertaTokenizer.tokenize(originalText);
    }
}
