// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.derived;

import com.yahoo.searchdefinition.Schema;
import com.yahoo.searchdefinition.NewApplicationBuilder;
import com.yahoo.searchdefinition.AbstractSchemaTestCase;
import com.yahoo.searchdefinition.parser.ParseException;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Correct casing for derived attributes
 *
 * @author vegardh
 */
public class CasingTestCase extends AbstractSchemaTestCase {

    @Test
    public void testCasing() throws IOException, ParseException {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/casing.sd");
        assertEquals(schema.getIndex("color").getName(), "color");
        assertEquals(schema.getIndex("Foo").getName(), "Foo");
        assertEquals(schema.getIndex("Price").getName(), "Price");
        assertEquals(schema.getAttribute("artist").getName(), "artist");
        assertEquals(schema.getAttribute("Drummer").getName(), "Drummer");
        assertEquals(schema.getAttribute("guitarist").getName(), "guitarist");
        assertEquals(schema.getAttribute("title").getName(), "title");
        assertEquals(schema.getAttribute("Trumpetist").getName(), "Trumpetist");
        assertEquals(schema.getAttribute("Saxophonist").getName(), "Saxophonist");
        assertEquals(schema.getAttribute("TenorSaxophonist").getName(), "TenorSaxophonist");
        assertEquals(schema.getAttribute("Flutist").getName(), "Flutist");
    }
}
