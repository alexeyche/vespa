// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.derived;

import com.yahoo.searchdefinition.NewApplicationBuilder;
import com.yahoo.searchdefinition.parser.ParseException;
import org.junit.Test;

import java.io.IOException;

/**
 * Test structs for streaming with another unrelated .sd present
 *
 * @author arnej27959
 */
public class TwoStreamingStructsTestCase extends AbstractExportingTestCase {

    @Test
    public void testTwoStreamingStructsExporting() throws ParseException, IOException {

        String root = "src/test/derived/twostreamingstructs";
        NewApplicationBuilder builder = new NewApplicationBuilder();
        builder.addSchemaFile(root + "/streamingstruct.sd");
        builder.addSchemaFile(root + "/whatever.sd");
        builder.build(true);
        assertCorrectDeriving(builder, builder.getSchema("streamingstruct"), root);

        builder = new NewApplicationBuilder();
        builder.addSchemaFile(root + "/streamingstruct.sd");
        builder.addSchemaFile(root + "/whatever.sd");
        builder.build(true);
        assertCorrectDeriving(builder, builder.getSchema("streamingstruct"), root);
    }

}
