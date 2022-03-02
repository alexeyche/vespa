// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition;

import com.yahoo.searchdefinition.parser.ParseException;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
/**
 * Tests settings outside the document
 *
 * @author  bratseth
 */
public class OutsideTestCase extends AbstractSchemaTestCase {

    @Test
    public void testOutsideIndex() throws IOException, ParseException {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/outsidedoc.sd");

        Index defaultIndex= schema.getIndex("default");
        assertTrue(defaultIndex.isPrefix());
        assertEquals("default.default",defaultIndex.aliasIterator().next());
    }

    @Test
    public void testOutsideSummary() throws IOException, ParseException {
        NewApplicationBuilder.buildFromFile("src/test/examples/outsidesummary.sd");
    }

}
