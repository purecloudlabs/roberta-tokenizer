package com.genesys.roberta.tokenizer;

import lombok.NonNull;

import java.io.FileNotFoundException;
import java.nio.file.Paths;

import static com.genesys.roberta.tokenizer.validation.Validator.checkPathExists;

/**
 * Creating a Resources Factory to enable lazy initialization
 */
public class RobertaTokenizerResourcesFactory {

    private final String baseDirPath;

    /**
     * Creates the Roberta Resource factory
     * @param baseDirPath directory where all the resource files are located
     * @throws FileNotFoundException if any of the resource files do not exist in the directory
     */
    public RobertaTokenizerResourcesFactory(final @NonNull String baseDirPath) throws FileNotFoundException {
        checkPathExists(Paths.get(baseDirPath),
                String.format("RobertaSentenceTokenizer base directory path: [%s] was not found", baseDirPath));
        this.baseDirPath = baseDirPath;
    }

    /**
     * This architecture allows us to keep the baseDirPath and create the RobertaTokenizerResources lazily i.e., only when needed.
     * @return a new RobertaTokenizerResources
     */
    public RobertaTokenizerResources create() {
        return new RobertaTokenizerResources(baseDirPath);
    }
}
