package com.genesys.roberta.tokenizer.validation;

import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Validator {

    /**
     * Validates the given path exists
     * @param path path in the file system
     * @param errorMsg error message that will be thrown/logged
     * @throws FileNotFoundException if the path does not exist
     */
    public static void checkPathExists(Path path, String errorMsg) throws FileNotFoundException {
        if (!Files.exists(path)) {
            throw new FileNotFoundException(errorMsg);
        }
    }
}
