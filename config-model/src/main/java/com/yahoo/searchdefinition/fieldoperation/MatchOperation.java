// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.fieldoperation;

import com.yahoo.searchdefinition.document.Case;
import com.yahoo.searchdefinition.document.MatchAlgorithm;
import com.yahoo.searchdefinition.document.Matching;
import com.yahoo.searchdefinition.document.MatchType;
import com.yahoo.searchdefinition.document.SDField;

/**
 * @author Einar M R Rosenvinge
 */
public class MatchOperation implements FieldOperation {

    private MatchType matchingType;
    private Case casing;
    private Integer gramSize;
    private Integer maxLength;
    private MatchAlgorithm matchingAlgorithm;
    private String exactMatchTerminator;

    public void setMatchingType(MatchType matchingType) {
        this.matchingType = matchingType;
    }

    public void setGramSize(Integer gramSize) {
        this.gramSize = gramSize;
    }
    public void setMaxLength(Integer maxLength) {
        this.maxLength = maxLength;
    }

    public void setMatchingAlgorithm(MatchAlgorithm matchingAlgorithm) {
        this.matchingAlgorithm = matchingAlgorithm;
    }

    public void setExactMatchTerminator(String exactMatchTerminator) {
        this.exactMatchTerminator = exactMatchTerminator;
    }

    public void setCase(Case casing) {
        this.casing = casing;
    }

    public void apply(SDField field) {
        if (matchingType != null) {
            field.setMatchingType(matchingType);
        }
        if (casing != null) {
            field.setMatchingCase(casing);
        }
        if (gramSize != null) {
            field.getMatching().setGramSize(gramSize);
        }
        if (maxLength != null) {
            field.getMatching().maxLength(maxLength);
        }
        if (matchingAlgorithm != null) {
            field.setMatchingAlgorithm(matchingAlgorithm);
        }
        if (exactMatchTerminator != null) {
            field.getMatching().setExactMatchTerminator(exactMatchTerminator);
        }
    }

}
