// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition;

import com.yahoo.searchdefinition.parser.ParseException;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertNotNull;

/**
 * @author bratseth
 */
public class ReservedWordsAsFieldNamesTestCase extends AbstractSchemaTestCase {

    @Test
    public void testIt() throws IOException, ParseException {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/reserved_words_as_field_names.sd");
        assertNotNull(schema.getDocument().getField("inline"));
        assertNotNull(schema.getDocument().getField("constants"));
        assertNotNull(schema.getDocument().getField("reference"));
    }

}
