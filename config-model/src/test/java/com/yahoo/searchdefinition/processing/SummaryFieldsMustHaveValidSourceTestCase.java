// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.processing;

import com.yahoo.config.model.application.provider.BaseDeployLogger;

import com.yahoo.searchdefinition.RankProfileRegistry;
import com.yahoo.searchdefinition.Schema;
import com.yahoo.searchdefinition.NewApplicationBuilder;
import com.yahoo.searchdefinition.AbstractSchemaTestCase;
import com.yahoo.searchdefinition.parser.ParseException;
import com.yahoo.vespa.model.container.search.QueryProfiles;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class SummaryFieldsMustHaveValidSourceTestCase extends AbstractSchemaTestCase {

    @Test
    public void requireThatInvalidSourceIsCaught() throws IOException, ParseException {
        try {
            NewApplicationBuilder.buildFromFile("src/test/examples/invalidsummarysource.sd");
            fail("This should throw and never get here");
        } catch (IllegalArgumentException e) {
            assertEquals("For schema 'invalidsummarysource', summary class 'baz', summary field 'cox': there is no valid source 'nonexistingfield'.", e.getMessage());
        }
    }

    @Test
    public void requireThatInvalidImplicitSourceIsCaught() throws IOException, ParseException {
        try {
            NewApplicationBuilder.buildFromFile("src/test/examples/invalidimplicitsummarysource.sd");
            fail("This should throw and never get here");
        } catch (IllegalArgumentException e) {
            assertEquals("For schema 'invalidsummarysource', summary class 'baz', summary field 'cox': there is no valid source 'cox'.", e.getMessage());
        }
    }

    @Test
    public void requireThatInvalidSelfReferingSingleSource() throws IOException, ParseException {
        try {
            NewApplicationBuilder.buildFromFile("src/test/examples/invalidselfreferringsummary.sd");
            fail("This should throw and never get here");
        } catch (IllegalArgumentException e) {
            assertEquals("For schema 'invalidselfreferringsummary', summary class 'withid', summary field 'w': there is no valid source 'w'.", e.getMessage());
        }
    }

    @Test
    public void requireThatDocumentIdIsAllowedToPass() throws IOException, ParseException {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/documentidinsummary.sd");
        BaseDeployLogger deployLogger = new BaseDeployLogger();
        RankProfileRegistry rankProfileRegistry = new RankProfileRegistry();
        new SummaryFieldsMustHaveValidSource(schema, deployLogger, rankProfileRegistry, new QueryProfiles()).process(true, false);
        assertEquals("documentid", schema.getSummary("withid").getSummaryField("w").getSingleSource());
    }

}
