package com.genesys.roberta.tokenizer.utils;

import com.genesys.roberta.tokenizer.pub.RobertaTokenizer;

import java.io.File;
import java.util.Objects;

public class CommonTestUtils {

    public static String getResourceAbsPath() {
        String resourceRelPath = "test-vocabularies";
        return new File(Objects.requireNonNull(RobertaTokenizer.class.getClassLoader().getResource(resourceRelPath))
                .getFile()).getAbsolutePath();
    }
}
