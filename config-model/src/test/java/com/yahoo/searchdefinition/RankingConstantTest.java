// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition;

import com.yahoo.searchdefinition.parser.ParseException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.Iterator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static com.yahoo.config.model.test.TestUtil.joinLines;
import static org.junit.Assert.fail;

/**
 * @author gjoranv
 */
public class RankingConstantTest {

    @SuppressWarnings("deprecation")
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Test
    public void tensor_constant_properties_are_set() throws Exception {
        final String TENSOR_NAME = "my_global_tensor";
        final String TENSOR_FILE = "path/my-tensor-file.json";
        final String TENSOR_TYPE = "tensor(x{})";
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  rank-profile my_rank_profile {",
                "    first-phase {",
                "      expression: sum(constant(my_global_tensor))",
                "    }",
                "  }",
                "  constant " + TENSOR_NAME + " {",
                "    file: " + TENSOR_FILE,
                "    type: " + TENSOR_TYPE,
                "  }",
                "}"
        ));
        schemaBuilder.build(true);
        Schema schema = schemaBuilder.getSchema();

        Iterator<RankingConstant> constantIterator = schema.rankingConstants().asMap().values().iterator();
        RankingConstant constant = constantIterator.next();
        assertEquals(TENSOR_NAME, constant.getName());
        assertEquals(TENSOR_FILE, constant.getFileName());
        assertEquals(TENSOR_TYPE, constant.getType());
        assertEquals(RankingConstant.PathType.FILE, constant.getPathType());

        assertFalse(constantIterator.hasNext());
    }

    @Test
    public void tensor_constant_must_have_a_type() throws Exception {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("must have a type");
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  constant foo {",
                "    file: bar.baz",
                "  }",
                "}"
        ));
    }

    @Test
    public void tensor_constant_must_have_a_file() throws Exception {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("must have a file");
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  constant foo {",
                "    type: tensor(x[])",
                "  }",
                "}"
        ));
    }

    @Test
    public void constant_file_does_not_need_path_or_ending() throws Exception {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  constant foo {",
                "    type: tensor(x{})",
                "    file: simplename",
                "  }",
                "}"
        ));
        schemaBuilder.build(true);
        Schema schema = schemaBuilder.getSchema();
        RankingConstant constant = schema.rankingConstants().asMap().values().iterator().next();
        assertEquals("simplename", constant.getFileName());
    }

    @Test
    public void constant_uri_is_allowed() throws Exception {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  constant foo {",
                "    type: tensor(x{})",
                "    uri: http://somewhere.far.away/in/another-galaxy",
                "  }",
                "}"
        ));
        schemaBuilder.build(true);
        Schema schema = schemaBuilder.getSchema();
        RankingConstant constant = schema.rankingConstants().asMap().values().iterator().next();
        assertEquals(RankingConstant.PathType.URI, constant.getPathType());
        assertEquals("http://somewhere.far.away/in/another-galaxy", constant.getUri());
    }

    @Test
    public void constant_https_uri_is_allowed() throws Exception {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  constant foo {",
                "    type: tensor(x{})",
                "    uri: https://somewhere.far.away:4443/in/another-galaxy",
                "  }",
                "}"
        ));
        schemaBuilder.build(true);
        Schema schema = schemaBuilder.getSchema();
        RankingConstant constant = schema.rankingConstants().asMap().values().iterator().next();
        assertEquals(RankingConstant.PathType.URI, constant.getPathType());
        assertEquals("https://somewhere.far.away:4443/in/another-galaxy", constant.getUri());
    }

    @Test
    public void constant_uri_with_port_is_allowed() throws Exception {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  constant foo {",
                "    type: tensor(x{})",
                "    uri: http://somewhere.far.away:4080/in/another-galaxy",
                "  }",
                "}"
        ));
        schemaBuilder.build(true);
        Schema schema = schemaBuilder.getSchema();
        RankingConstant constant = schema.rankingConstants().asMap().values().iterator().next();
        assertEquals(RankingConstant.PathType.URI, constant.getPathType());
        assertEquals("http://somewhere.far.away:4080/in/another-galaxy", constant.getUri());
    }

    @Test
    public void constant_uri_no_dual_slashes_is_allowed() throws Exception {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        schemaBuilder.addSchema(joinLines(
                "search test {",
                "  document test { }",
                "  constant foo {",
                "    type: tensor(x{})",
                "    uri: http:somewhere.far.away/in/another-galaxy",
                "  }",
                "}"
        ));
        schemaBuilder.build(true);
        Schema schema = schemaBuilder.getSchema();
        RankingConstant constant = schema.rankingConstants().asMap().values().iterator().next();
        assertEquals(RankingConstant.PathType.URI, constant.getPathType());
        assertEquals("http:somewhere.far.away/in/another-galaxy", constant.getUri());
    }

    @Test
    public void constant_uri_only_supports_http_and_https() {
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        NewApplicationBuilder schemaBuilder = new NewApplicationBuilder(rankProfileRegistry);
        String expectedMessage = "Encountered \" <IDENTIFIER> \"ftp\"\" at line 5, column 10.\n\n" +
                "Was expecting:\n\n" +
                "<URI_PATH> ...";
        try {
            schemaBuilder.addSchema(joinLines(
                    "search test {",
                    "  document test { }",
                    "  constant foo {",
                    "    type: tensor(x{})",
                    "    uri: ftp:somewhere.far.away/in/another-galaxy",
                    "  }",
                    "}"
            ));
        } catch (ParseException e) {
            if (! e.getMessage().startsWith(expectedMessage))
                fail("Expected exception with message starting with:\n'" + expectedMessage + "\nBut got:\n'" + e.getMessage());
        }
    }

}
