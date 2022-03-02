// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition;

import com.yahoo.document.ArrayDataType;
import com.yahoo.document.CollectionDataType;
import com.yahoo.document.DataType;
import com.yahoo.searchdefinition.document.SDField;
import com.yahoo.searchdefinition.parser.ParseException;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * tests importing of document containing array type fields
 *
 * @author bratseth
 */
public class ArraysTestCase extends AbstractSchemaTestCase {

    @Test
    public void testArrayImporting() throws IOException, ParseException {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/arrays.sd");

        SDField tags = (SDField) schema.getDocument().getField("tags");
        assertEquals(DataType.STRING, ((CollectionDataType)tags.getDataType()).getNestedType());

        SDField ratings = (SDField) schema.getDocument().getField("ratings");
        assertTrue(ratings.getDataType() instanceof ArrayDataType);
        assertEquals(DataType.INT, ((ArrayDataType)ratings.getDataType()).getNestedType());
    }

}
