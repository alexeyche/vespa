// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.parser;

import com.yahoo.searchdefinition.RankProfile;
import com.yahoo.searchdefinition.RankProfileRegistry;
import com.yahoo.searchdefinition.Schema;
import com.yahoo.searchdefinition.document.RankType;
import com.yahoo.searchlib.rankingexpression.evaluation.TensorValue;
import com.yahoo.searchlib.rankingexpression.evaluation.Value;

import java.util.List;

/**
 * Helper for converting ParsedRankProfile etc to RankProfile with settings
 *
 * @author arnej27959
 **/
public class ConvertParsedRanking {

    private final RankProfileRegistry rankProfileRegistry;

    // for unit test
    ConvertParsedRanking() {
        this(new RankProfileRegistry());
    }

    public ConvertParsedRanking(RankProfileRegistry rankProfileRegistry) {
        this.rankProfileRegistry = rankProfileRegistry;
    }

    private RankProfile makeRankProfile(Schema schema, String name) {
        if (name.equals("default")) {
            return rankProfileRegistry.get(schema, "default");
        }
        return new RankProfile(name, schema, rankProfileRegistry, schema.rankingConstants());
    }

    void convertRankProfile(Schema schema, ParsedRankProfile parsed) {
        final RankProfile profile = makeRankProfile(schema, parsed.name());
        assert(profile != null);
        for (String name : parsed.getInherited()) {
            profile.inherit(name);
        }
        parsed.isStrict().ifPresent(value -> profile.setStrict(value));

        for (var entry : parsed.getConstants().entrySet()) {
            String name = entry.getKey();
            Value value = entry.getValue();
            if (value instanceof TensorValue) {
                var tensor = (TensorValue) value;
                profile.addConstantTensor(name, tensor);
            } else {
                profile.addConstant(name, value);
            }
        }

        for (var input : parsed.getInputs().entrySet()) {
            profile.addInput(input.getKey(), input.getValue());
        }

        for (var func : parsed.getFunctions()) {
            String name = func.name();
            List<String> parameters = func.getParameters();
            String expression = func.getExpression();
            boolean inline = func.getInline();
            profile.addFunction(name, parameters, expression, inline);
        }

        parsed.getRankScoreDropLimit().ifPresent
            (value -> profile.setRankScoreDropLimit(value));
        parsed.getTermwiseLimit().ifPresent
            (value -> profile.setTermwiseLimit(value));
        parsed.getPostFilterThreshold().ifPresent
                (value -> profile.setPostFilterThreshold(value));
        parsed.getApproximateThreshold().ifPresent
                (value -> profile.setApproximateThreshold(value));
        parsed.getKeepRankCount().ifPresent
            (value -> profile.setKeepRankCount(value));
        parsed.getMinHitsPerThread().ifPresent
            (value -> profile.setMinHitsPerThread(value));
        parsed.getNumSearchPartitions().ifPresent
            (value -> profile.setNumSearchPartitions(value));
        parsed.getNumThreadsPerSearch().ifPresent
            (value -> profile.setNumThreadsPerSearch(value));
        parsed.getReRankCount().ifPresent
            (value -> profile.setRerankCount(value));

        parsed.getMatchPhaseSettings().ifPresent
            (value -> profile.setMatchPhaseSettings(value));

        parsed.getFirstPhaseExpression().ifPresent
            (value -> profile.setFirstPhaseRanking(value));
        parsed.getSecondPhaseExpression().ifPresent
            (value -> profile.setSecondPhaseRanking(value));

        for (var value : parsed.getMatchFeatures()) {
            profile.addMatchFeatures(value);
        }
        for (var value : parsed.getRankFeatures()) {
            profile.addRankFeatures(value);
        }
        for (var value : parsed.getSummaryFeatures()) {
            profile.addSummaryFeatures(value);
        }

        parsed.getInheritedMatchFeatures().ifPresent
            (value -> profile.setInheritedMatchFeatures(value));
        parsed.getInheritedSummaryFeatures().ifPresent
            (value -> profile.setInheritedSummaryFeatures(value));
        if (parsed.getIgnoreDefaultRankFeatures()) {
            profile.setIgnoreDefaultRankFeatures(true);
        }

        for (var mutateOp : parsed.getMutateOperations()) {
            profile.addMutateOperation(mutateOp);
        }
        parsed.getFieldsWithRankFilter().forEach
            ((fieldName, isFilter) -> profile.addRankSetting(fieldName, RankProfile.RankSetting.Type.PREFERBITVECTOR, isFilter));

        parsed.getFieldsWithRankWeight().forEach
            ((fieldName, weight) -> profile.addRankSetting(fieldName, RankProfile.RankSetting.Type.WEIGHT, weight));

        parsed.getFieldsWithRankType().forEach
            ((fieldName, rankType) -> profile.addRankSetting(fieldName, RankProfile.RankSetting.Type.RANKTYPE, RankType.fromString(rankType)));

        parsed.getRankProperties().forEach
            ((key, values) -> {for (String value : values) profile.addRankProperty(key, value);});

        // always?
        rankProfileRegistry.add(profile);
    }

}
