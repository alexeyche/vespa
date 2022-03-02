// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.processing;

import com.yahoo.document.DataType;
import com.yahoo.document.DocumentType;
import com.yahoo.document.PositionDataType;
import com.yahoo.searchdefinition.Schema;
import com.yahoo.searchdefinition.NewApplicationBuilder;
import com.yahoo.searchdefinition.document.Attribute;
import com.yahoo.searchdefinition.document.FieldSet;
import com.yahoo.vespa.documentmodel.SummaryField;
import com.yahoo.vespa.documentmodel.SummaryTransform;

import org.junit.Test;

import java.util.Arrays;
import java.util.Iterator;

import static org.junit.Assert.*;

/**
 * Test Position processor.
 *
 * @author hmusum
 */
public class PositionTestCase {

    @Test
    public void inherited_position_zcurve_field_is_not_added_to_document_fieldset() throws Exception {
        NewApplicationBuilder sb = NewApplicationBuilder.createFromFiles(Arrays.asList(
                "src/test/examples/position_base.sd",
                "src/test/examples/position_inherited.sd"));

        Schema schema = sb.getSchema("position_inherited");
        FieldSet fieldSet = schema.getDocument().getFieldSets().builtInFieldSets().get(DocumentType.DOCUMENT);
        assertFalse(fieldSet.getFieldNames().contains(PositionDataType.getZCurveFieldName("pos")));
    }

    @Test
    public void requireThatPositionCanBeAttribute() throws Exception {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/position_attribute.sd");
        assertNull(schema.getAttribute("pos"));
        assertNull(schema.getAttribute("pos.x"));
        assertNull(schema.getAttribute("pos.y"));

        assertPositionAttribute(schema, "pos", Attribute.CollectionType.SINGLE);
        assertPositionSummary(schema, "pos", false);
    }

    @Test
    public void requireThatPositionCanNotBeIndex() throws Exception {
        try {
            NewApplicationBuilder.buildFromFile("src/test/examples/position_index.sd");
            fail();
        } catch (IllegalArgumentException e) {
            assertEquals("For schema 'position_index', field 'pos': Indexing of data type 'position' is not " +
                         "supported, replace 'index' statement with 'attribute'.", e.getMessage());
        }
    }

    @Test
    public void requireThatSummaryAloneDoesNotCreateZCurve() throws Exception {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/position_summary.sd");
        assertNull(schema.getAttribute("pos"));
        assertNull(schema.getAttribute("pos.x"));
        assertNull(schema.getAttribute("pos.y"));
        assertNull(schema.getAttribute("pos.zcurve"));

        SummaryField summary = schema.getSummaryField("pos");
        assertNotNull(summary);
        assertEquals(2, summary.getSourceCount());
        Iterator<SummaryField.Source> it = summary.getSources().iterator();
        assertEquals("pos.x", it.next().getName());
        assertEquals("pos.y", it.next().getName());
        assertEquals(SummaryTransform.NONE, summary.getTransform());

        assertNull(schema.getSummaryField("pos_ext.distance"));
    }

    @Test
    public void requireThatExtraFieldCanBePositionAttribute() throws Exception {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/position_extra.sd");
        assertNull(schema.getAttribute("pos_ext"));
        assertNull(schema.getAttribute("pos_ext.x"));
        assertNull(schema.getAttribute("pos_ext.y"));

        assertPositionAttribute(schema, "pos_ext", Attribute.CollectionType.SINGLE);
        assertPositionSummary(schema, "pos_ext", false);
    }

    @Test
    public void requireThatPositionArrayIsSupported() throws Exception {
        Schema schema = NewApplicationBuilder.buildFromFile("src/test/examples/position_array.sd");
        assertNull(schema.getAttribute("pos"));
        assertNull(schema.getAttribute("pos.x"));
        assertNull(schema.getAttribute("pos.y"));

        assertPositionAttribute(schema, "pos", Attribute.CollectionType.ARRAY);
        assertPositionSummary(schema, "pos", true);
    }

    private static void assertPositionAttribute(Schema schema, String fieldName, Attribute.CollectionType type) {
        Attribute attribute = schema.getAttribute(PositionDataType.getZCurveFieldName(fieldName));
        assertNotNull(attribute);
        assertTrue(attribute.isPosition());
        assertEquals(attribute.getCollectionType(), type);
        assertEquals(attribute.getType(), Attribute.Type.LONG);
    }

    private static void assertPositionSummary(Schema schema, String fieldName, boolean isArray) {
        assertSummaryField(schema,
                           fieldName,
                           PositionDataType.getZCurveFieldName(fieldName),
                           (isArray ? DataType.getArray(PositionDataType.INSTANCE) : PositionDataType.INSTANCE),
                           SummaryTransform.GEOPOS);
        assertSummaryField(schema,
                           PositionDataType.getDistanceSummaryFieldName(fieldName),
                           PositionDataType.getZCurveFieldName(fieldName),
                           DataType.INT,
                           SummaryTransform.DISTANCE);
        assertSummaryField(schema,
                           PositionDataType.getPositionSummaryFieldName(fieldName),
                           PositionDataType.getZCurveFieldName(fieldName),
                           DataType.getArray(DataType.STRING),
                           SummaryTransform.POSITIONS);
    }

    private static void assertSummaryField(Schema schema, String fieldName, String sourceName, DataType dataType,
                                           SummaryTransform transform)
    {
        SummaryField summary = schema.getSummaryField(fieldName);
        assertNotNull(summary);
        assertEquals(1, summary.getSourceCount());
        assertEquals(sourceName, summary.getSingleSource());
        assertEquals(dataType, summary.getDataType());
        assertEquals(transform, summary.getTransform());
    }
}
