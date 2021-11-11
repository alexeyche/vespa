// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition;

import com.yahoo.collections.Pair;
import com.yahoo.config.application.api.DeployLogger;
import com.yahoo.config.model.application.provider.MockFileRegistry;
import com.yahoo.config.model.deploy.TestProperties;
import com.yahoo.config.model.test.MockApplicationPackage;
import com.yahoo.search.query.profile.QueryProfileRegistry;
import com.yahoo.searchdefinition.derived.AttributeFields;
import com.yahoo.searchdefinition.derived.RawRankProfile;
import com.yahoo.searchdefinition.parser.ParseException;
import ai.vespa.rankingexpression.importer.configmodelview.ImportedMlModels;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Optional;
import java.util.logging.Level;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author bratseth
 */
public class RankingExpressionInliningTestCase extends AbstractSchemaTestCase {

    @Test
    public void testFunctionInliningPreserveArithmeticOrdering() throws ParseException {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        SchemaBuilder builder = new SchemaBuilder(rankProfileRegistry);
        builder.importString(
                        "search test {\n" +
                        "    document test { \n" +
                        "        field a type double { \n" +
                        "            indexing: attribute \n" +
                        "        }\n" +
                        "        field b type double { \n" +
                        "            indexing: attribute \n" +
                        "        }\n" +
                        "    }\n" +
                        "    \n" +
                        "    rank-profile parent {\n" +
                        "        constants {\n" +
                        "            p1: 7 \n" +
                        "            p2: 0 \n" +
                        "        }\n" +
                        "        first-phase {\n" +
                        "            expression: p1 * add\n" +
                        "        }\n" +
                        "        function inline add() {\n" +
                        "            expression: 3 + attribute(a) + attribute(b) * mul3\n" +
                        "        }\n" +
                        "        function inline mul3() {\n" +
                        "            expression: attribute(a) * 3 + singleif\n" +
                        "        }\n" +
                        "        function inline singleif() {\n" +
                        "            expression: if (p1 < attribute(a), 1, 2) == 0\n" +
                        "        }\n" +
                        "    }\n" +
                        "    rank-profile child inherits parent {\n" +
                        "        function inline add() {\n" +
                        "            expression: 9 + attribute(a)\n" +
                        "        }\n" +
                        "    }\n" +
                        "\n" +
                        "}\n");
        builder.build();
        Schema s = builder.getSchema();

        RankProfile parent = rankProfileRegistry.get(s, "parent").compile(new QueryProfileRegistry(), new ImportedMlModels());
        assertEquals("7 * (3 + attribute(a) + attribute(b) * (attribute(a) * 3 + if (7 < attribute(a), 1, 2) == 0))",
                     parent.getFirstPhaseRanking().getRoot().toString());
        RankProfile child = rankProfileRegistry.get(s, "child").compile(new QueryProfileRegistry(), new ImportedMlModels());
        assertEquals("7 * (9 + attribute(a))",
                     child.getFirstPhaseRanking().getRoot().toString());
    }

    @Test
    public void testConstants() throws ParseException {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        SchemaBuilder builder = new SchemaBuilder(rankProfileRegistry);
        builder.importString(
                "search test {\n" +
                        "    document test { \n" +
                        "        field a type string { \n" +
                        "            indexing: index \n" +
                        "        }\n" +
                        "    }\n" +
                        "    \n" +
                        "    rank-profile parent {\n" +
                        "        constants {\n" +
                        "            p1: 7 \n" +
                        "            p2: 0 \n" +
                        "        }\n" +
                        "        first-phase {\n" +
                        "            expression: p1 + foo\n" +
                        "        }\n" +
                        "        second-phase {\n" +
                        "            expression: p2 * foo\n" +
                        "        }\n" +
                        "        function inline foo() {\n" +
                        "            expression: 3 + p1 + p2\n" +
                        "        }\n" +
                        "    }\n" +
                        "    rank-profile child inherits parent {\n" +
                        "        first-phase {\n" +
                        "            expression: p1 + foo + baz + bar + arg(4.0)\n" +
                        "        }\n" +
                        "        constants {\n" +
                        "            p2: 2.0 \n" +
                        "        }\n" +
                        "        function bar() {\n" +
                        "            expression: p2*p1\n" +
                        "        }\n" +
                        "        function inline baz() {\n" +
                        "            expression: p2+p1+boz\n" +
                        "        }\n" +
                        "        function inline boz() {\n" +
                        "            expression: 3.0\n" +
                        "        }\n" +
                        "        function inline arg(a1) {\n" +
                        "            expression: a1*2\n" +
                        "        }\n" +
                        "    }\n" +
                        "\n" +
                        "}\n");
        builder.build();
        Schema s = builder.getSchema();

        RankProfile parent = rankProfileRegistry.get(s, "parent").compile(new QueryProfileRegistry(), new ImportedMlModels());
        assertEquals("17.0", parent.getFirstPhaseRanking().getRoot().toString());
        assertEquals("0.0", parent.getSecondPhaseRanking().getRoot().toString());
        assertEquals("10.0", getRankingExpression("foo", parent, s));
        assertEquals("17.0", getRankingExpression("firstphase", parent, s));
        assertEquals("0.0", getRankingExpression("secondphase", parent, s));

        RankProfile child = rankProfileRegistry.get(s, "child").compile(new QueryProfileRegistry(), new ImportedMlModels());
        assertEquals("31.0 + bar + arg(4.0)", child.getFirstPhaseRanking().getRoot().toString());
        assertEquals("24.0", child.getSecondPhaseRanking().getRoot().toString());
        assertEquals("12.0", getRankingExpression("foo", child, s));
        assertEquals("12.0", getRankingExpression("baz", child, s));
        assertEquals("3.0", getRankingExpression("boz", child, s));
        assertEquals("14.0", getRankingExpression("bar", child, s));
        assertEquals("a1 * 2", getRankingExpression("arg", child, s));
        assertEquals("31.0 + rankingExpression(bar) + rankingExpression(arg@)", getRankingExpression("firstphase", child, s));
        assertEquals("24.0", getRankingExpression("secondphase", child, s));
    }

    @Test
    public void testNonTopLevelInlining() throws ParseException {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        SchemaBuilder builder = new SchemaBuilder(rankProfileRegistry);
        builder.importString(
                "search test {\n" +
                        "    document test { \n" +
                        "        field a type double { \n" +
                        "            indexing: attribute \n" +
                        "        }\n" +
                        "        field b type double { \n" +
                        "            indexing: attribute \n" +
                        "        }\n" +
                        "    }\n" +
                        "    \n" +
                        "    rank-profile test {\n" +
                        "        first-phase {\n" +
                        "            expression: A + C + D\n" +
                        "        }\n" +
                        "        function inline D() {\n" +
                        "            expression: B + 1\n" +
                        "        }\n" +
                        "        function C() {\n" +
                        "            expression: A + B\n" +
                        "        }\n" +
                        "        function inline B() {\n" +
                        "            expression: attribute(b)\n" +
                        "        }\n" +
                        "        function inline A() {\n" +
                        "            expression: attribute(a)\n" +
                        "        }\n" +
                        "    }\n" +
                        "\n" +
                        "}\n");
        builder.build();
        Schema s = builder.getSchema();

        RankProfile test = rankProfileRegistry.get(s, "test").compile(new QueryProfileRegistry(), new ImportedMlModels());
        assertEquals("attribute(a) + C + (attribute(b) + 1)", test.getFirstPhaseRanking().getRoot().toString());
        assertEquals("attribute(a) + attribute(b)", getRankingExpression("C", test, s));
        assertEquals("attribute(b) + 1", getRankingExpression("D", test, s));
    }

    @Test
    public void testFunctionInliningWithReplacement() throws ParseException {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        MockDeployLogger deployLogger = new MockDeployLogger();
        SchemaBuilder builder = new SchemaBuilder(MockApplicationPackage.createEmpty(),
                                                  new MockFileRegistry(),
                                                  deployLogger,
                                                  new TestProperties(),
                                                  rankProfileRegistry,
                                                  new QueryProfileRegistry());
        builder.importString(
                        "search test {\n" +
                        "    document test { }\n" +
                        "    rank-profile test {\n" +
                        "        first-phase {\n" +
                        "            expression: foo\n" +
                        "        }\n" +
                        "        function foo(x) {\n" +
                        "            expression: x + x\n" +
                        "        }\n" +
                        "        function inline foo() {\n" +  // replaces previous "foo" during parsing
                        "            expression: foo(2)\n" +
                        "        }\n" +
                        "    }\n" +
                        "}\n");
        builder.build();
        Schema s = builder.getSchema();
        RankProfile test = rankProfileRegistry.get(s, "test").compile(new QueryProfileRegistry(), new ImportedMlModels());
        assertEquals("foo(2)", test.getFirstPhaseRanking().getRoot().toString());
        assertTrue("Does not contain expected warning", deployLogger.contains("Function 'foo' replaces " +
                "a previous function with the same name in rank profile 'test'"));
    }

    /**
     * Expression evaluation has no stack so function arguments are bound at config time creating a separate version of
     * each function for each binding, using hashes to name the bound variants of the function.
     * This method censors those hashes for string comparison.
     */
    private String censorBindingHash(String s) {
        StringBuilder b = new StringBuilder();
        boolean areInHash = false;
        for (int i = 0; i < s.length() ; i++) {
            char current = s.charAt(i);

            if ( ! Character.isLetterOrDigit(current)) // end of hash
                areInHash = false;

            if ( ! areInHash)
                b.append(current);

            if (current == '@') // start of hash
                areInHash = true;
        }
        return b.toString();
    }

    private String getRankingExpression(String name, RankProfile rankProfile, Schema schema) {
        Optional<String> rankExpression =
                new RawRankProfile(rankProfile, new LargeRankExpressions(new MockFileRegistry()), new QueryProfileRegistry(), new ImportedMlModels(), new AttributeFields(schema), new TestProperties())
                        .configProperties()
                        .stream()
                        .filter(r -> r.getFirst().equals("rankingExpression(" + name + ").rankingScript"))
                        .map(Pair::getSecond)
                        .findFirst();
        assertTrue(rankExpression.isPresent());
        return censorBindingHash(rankExpression.get());
    }

    private static class MockDeployLogger implements DeployLogger {
        private final ArrayList<String> msgs = new ArrayList<>();

        @Override
        public void log(Level level, String message) {
            msgs.add(message);
        }

        public boolean contains(String expected) {
            return msgs.stream().anyMatch(msg -> msg.equals(expected));
        }
    }

}
