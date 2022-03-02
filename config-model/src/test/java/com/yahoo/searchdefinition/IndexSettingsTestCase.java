// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition;

import com.yahoo.searchdefinition.document.SDField;
import com.yahoo.searchdefinition.document.Stemming;
import com.yahoo.searchdefinition.parser.ParseException;
import org.junit.Test;

import java.io.IOException;

import static com.yahoo.config.model.test.TestUtil.joinLines;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Rank settings
 *
 * @author bratseth
 */
public class IndexSettingsTestCase extends AbstractSchemaTestCase {

    @Test
    public void testStemmingSettings() throws IOException, ParseException {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/indexsettings.sd");

        SDField usingDefault=(SDField) schema.getDocument().getField("usingdefault");
        assertEquals(Stemming.SHORTEST,usingDefault.getStemming(schema));

        SDField notStemmed=(SDField) schema.getDocument().getField("notstemmed");
        assertEquals(Stemming.NONE,notStemmed.getStemming(schema));

        SDField allStemmed=(SDField) schema.getDocument().getField("allstemmed");
        assertEquals(Stemming.SHORTEST,allStemmed.getStemming(schema));

        SDField multiStemmed=(SDField) schema.getDocument().getField("multiplestems");
        assertEquals(Stemming.MULTIPLE, multiStemmed.getStemming(schema));
    }

    @Test
    public void requireThatInterlavedFeaturesAreSetOnExtraField() throws ParseException {
        NewApplicationBuilder builder = NewApplicationBuilder.createFromString(joinLines(
                "search test {",
                "  document test {",
                "    field content type string {",
                "      indexing: index | summary",
                "      index: enable-bm25",
                "    }",
                "  }",
                "  field extra type string {",
                "    indexing: input content | index | summary",
                "    index: enable-bm25",
                "  }",
                "}"
        ));
        Schema schema = builder.getSchema();
        Index contentIndex = schema.getIndex("content");
        assertTrue(contentIndex.useInterleavedFeatures());
        Index extraIndex = schema.getIndex("extra");
        assertTrue(extraIndex.useInterleavedFeatures());
    }

}
