// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.parser;

import java.util.ArrayList;
import java.util.List;

/**
 * This class holds the extracted information after parsing a
 * "document" block in a schema (.sd) file, using simple data
 * structures as far as possible.  Do not put advanced logic here!
 * @author arnej27959
 **/
public class ParsedDocument {
    private final String name;
    private final List<String> inherited = new ArrayList<>();

    public  ParsedDocument(String name) {
        this.name = name;
    }

    String name() { return name; }
    void inherit(String other) { inherited.add(other); }

    void addField(ParsedField field) {}
    void addStruct(ParsedStruct type) {}
    void addAnnotation(ParsedAnnotation type) {}

    /*
    private final List<ParsedField> fields = new ArrayList<>();
    List<ParsedField> getFields() { return fields; }
    */
}

