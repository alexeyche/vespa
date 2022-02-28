// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition.parser;


import com.yahoo.config.application.api.ApplicationPackage;
import com.yahoo.config.application.api.DeployLogger;
import com.yahoo.config.application.api.FileRegistry;
import com.yahoo.config.model.api.ModelContext;
import com.yahoo.config.model.test.MockApplicationPackage;
import com.yahoo.document.DataType;
import com.yahoo.document.DataTypeName;
import com.yahoo.document.DocumentType;
import com.yahoo.document.DocumentTypeManager;
import com.yahoo.document.PositionDataType;
import com.yahoo.document.ReferenceDataType;
import com.yahoo.document.StructDataType;
import com.yahoo.document.WeightedSetDataType;
import com.yahoo.document.annotation.AnnotationReferenceDataType;
import com.yahoo.document.annotation.AnnotationType;
import com.yahoo.language.Linguistics;
import com.yahoo.language.process.Embedder;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.query.ranking.Diversity;
import com.yahoo.searchdefinition.DefaultRankProfile;
import com.yahoo.searchdefinition.DocumentOnlySchema;
import com.yahoo.searchdefinition.DocumentsOnlyRankProfile;
import com.yahoo.searchdefinition.Index;
import com.yahoo.searchdefinition.OnnxModel;
import com.yahoo.searchdefinition.RankProfile.DiversitySettings;
import com.yahoo.searchdefinition.RankProfile.MatchPhaseSettings;
import com.yahoo.searchdefinition.RankProfile;
import com.yahoo.searchdefinition.RankProfileRegistry;
import com.yahoo.searchdefinition.RankingConstant;
import com.yahoo.searchdefinition.Schema;
import com.yahoo.searchdefinition.UnrankedRankProfile;
import com.yahoo.searchdefinition.document.Attribute;
import com.yahoo.searchdefinition.document.SDDocumentType;
import com.yahoo.searchdefinition.document.SDField;
import com.yahoo.searchdefinition.document.Sorting;
import com.yahoo.searchdefinition.document.Stemming;
import com.yahoo.searchlib.rankingexpression.FeatureList;
import com.yahoo.searchlib.rankingexpression.evaluation.TensorValue;
import com.yahoo.searchlib.rankingexpression.evaluation.Value;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.vespa.documentmodel.DocumentSummary;
import com.yahoo.vespa.documentmodel.SummaryField;
import com.yahoo.vespa.documentmodel.SummaryTransform;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;

/**
 * Class converting a collection of schemas from the intermediate format.
 * For now only conversion to DocumentType (with contents).
 *
 * @author arnej27959
 **/
public class ConvertSchemaCollection {

    private final IntermediateCollection input;
    private final List<ParsedSchema> orderedInput = new ArrayList<>();
    private final DocumentTypeManager docMan;
    private final ApplicationPackage applicationPackage;
    private final FileRegistry fileRegistry;
    private final DeployLogger deployLogger;
    private final ModelContext.Properties properties;
    private final RankProfileRegistry rankProfileRegistry;
    private final boolean documentsOnly;

    // for unit test
    ConvertSchemaCollection(IntermediateCollection input,
                            DocumentTypeManager documentTypeManager)
    {
        this(input, documentTypeManager,
             null, null, null, null, null,
             true);
    }

    public ConvertSchemaCollection(IntermediateCollection input,
                                   DocumentTypeManager documentTypeManager,
                                   ApplicationPackage applicationPackage,
                                   FileRegistry fileRegistry,
                                   DeployLogger deployLogger,
                                   ModelContext.Properties properties,
                                   RankProfileRegistry rankProfileRegistry,
                                   boolean documentsOnly)
    {
        this.input = input;
        this.docMan = documentTypeManager;
        this.applicationPackage = applicationPackage;
        this.fileRegistry = fileRegistry;
        this.deployLogger = deployLogger;
        this.properties = properties;
        this.rankProfileRegistry = rankProfileRegistry;
        this.documentsOnly = documentsOnly;

        input.resolveInternalConnections();
        order();
        pushTypesToDocuments();
        convertDataTypes();
    }

    void order() {
        var map = input.getParsedSchemas();
        for (var schema : map.values()) {
            findOrdering(schema);
        }
    }

    void findOrdering(ParsedSchema schema) {
        if (orderedInput.contains(schema)) return;
        for (var parent : schema.getAllResolvedInherits()) {
            findOrdering(parent);
        }
        orderedInput.add(schema);
    }

    void pushTypesToDocuments() {
        for (var schema : orderedInput) {
            for (var struct : schema.getStructs()) {
                schema.getDocument().addStruct(struct);
            }
            for (var annotation : schema.getAnnotations()) {
                schema.getDocument().addAnnotation(annotation);
            }
        }
    }

    Map<String, DocumentType> documentsInProgress = new HashMap<>();
    Map<String, StructDataType> structsInProgress = new HashMap<>();
    Map<String, AnnotationType> annotationsInProgress = new HashMap<>();

    StructDataType findStructInProgress(String name, ParsedDocument context) {
        var resolved = findStructFrom(context, name);
        if (resolved == null) {
            throw new IllegalArgumentException("no struct named " + name + " in context " + context);
        }
        String structId = resolved.getOwner() + "->" + resolved.name();
        var struct = structsInProgress.get(structId);
        assert(struct != null);
        return struct;
    }

    AnnotationType findAnnotationInProgress(String name, ParsedDocument context) {
        var resolved = findAnnotationFrom(context, name);
        String annotationId = resolved.getOwner() + "->" + resolved.name();
        var annotation = annotationsInProgress.get(annotationId);
        if (annotation == null) {
            throw new IllegalArgumentException("no annotation named " + name + " in context " + context);
        }
        return annotation;
    }

    ParsedStruct findStructFrom(ParsedDocument doc, String name) {
        ParsedStruct found = doc.getStruct(name);
        if (found != null) return found;
        for (var parent : doc.getResolvedInherits()) {
            var fromParent = findStructFrom(parent, name);
            if (fromParent == null) continue;
            if (fromParent == found) continue;
            if (found == null) {
                found = fromParent;
            } else {
                throw new IllegalArgumentException("conflicting values for struct " + name + " in " +doc);
            }
        }
        return found;
    }

    ParsedAnnotation findAnnotationFrom(ParsedDocument doc, String name) {
        ParsedAnnotation found = doc.getAnnotation(name);
        if (found != null) return found;
        for (var parent : doc.getResolvedInherits()) {
            var fromParent = findAnnotationFrom(parent, name);
            if (fromParent == null) continue;
            if (fromParent == found) continue;
            if (found == null) {
                found = fromParent;
            } else {
                throw new IllegalArgumentException("conflicting values for annotation " + name + " in " +doc);
            }
        }
        return found;
    }

    private DataType createArray(ParsedType pType, ParsedDocument context) {
        DataType nested = resolveType(pType.nestedType(), context);
        return DataType.getArray(nested);
    }

    private DataType createWset(ParsedType pType, ParsedDocument context) {
        DataType nested = resolveType(pType.nestedType(), context);
        boolean cine = pType.getCreateIfNonExistent();
        boolean riz = pType.getRemoveIfZero();
        return new WeightedSetDataType(nested, cine, riz);
    }

    private DataType createMap(ParsedType pType, ParsedDocument context) {
        DataType kt = resolveType(pType.mapKeyType(), context);
        DataType vt = resolveType(pType.mapValueType(), context);
        return DataType.getMap(kt, vt);
    }

    private DocumentType findDocInProgress(String name) {
        var dt = documentsInProgress.get(name);
        if (dt == null) {
            throw new IllegalArgumentException("missing document type for: " + name);
        }
        return dt;
    }

    private DataType createAnnRef(ParsedType pType, ParsedDocument context) {
        AnnotationType annotation = findAnnotationInProgress(pType.getNameOfReferencedAnnotation(), context);
        return new AnnotationReferenceDataType(annotation);
    }

    private DataType createDocRef(ParsedType pType) {
        var ref = pType.getReferencedDocumentType();
        assert(ref.getVariant() == ParsedType.Variant.DOCUMENT);
        return ReferenceDataType.createWithInferredId(findDocInProgress(ref.name()));
    }

    DataType resolveType(ParsedType pType, ParsedDocument context) {
        switch (pType.getVariant()) {
        case NONE:     return DataType.NONE;
        case BUILTIN:  return docMan.getDataType(pType.name());
        case POSITION: return PositionDataType.INSTANCE;
        case ARRAY:    return createArray(pType, context);
        case WSET:     return createWset(pType, context);
        case MAP:      return createMap(pType, context);
        case TENSOR:   return DataType.getTensor(pType.getTensorType());
        case DOC_REFERENCE:  return createDocRef(pType);
        case ANN_REFERENCE:  return createAnnRef(pType, context);
        case DOCUMENT: return findDocInProgress(pType.name());
        case STRUCT:   return findStructInProgress(pType.name(), context);
        case UNKNOWN:
            // fallthrough
        }
        // unknown is probably struct, but could be document:
        if (documentsInProgress.containsKey(pType.name())) {
            pType.setVariant(ParsedType.Variant.DOCUMENT);
            return findDocInProgress(pType.name());
        }
        var struct = findStructInProgress(pType.name(), context);
        pType.setVariant(ParsedType.Variant.STRUCT);
        return struct;
    }

    void convertDataTypes() {
        for (var schema : orderedInput) {
            String name = schema.getDocument().name();
            documentsInProgress.put(name, new DocumentType(name));
        }
        for (var schema : orderedInput) {
            var doc = schema.getDocument();
            for (var struct : doc.getStructs()) {
                var dt = new StructDataType(struct.name());
                String structId = doc.name() + "->" + struct.name();
                structsInProgress.put(structId, dt);
            }
            for (var annotation : doc.getAnnotations()) {
                String annId = doc.name() + "->" + annotation.name();
                var at = new AnnotationType(annotation.name());
                annotationsInProgress.put(annId, at);
                var withStruct = annotation.getStruct();
                if (withStruct.isPresent()) {
                    var sn = withStruct.get().name();
                    var dt = new StructDataType(sn);
                    String structId = doc.name() + "->" + sn;
                    structsInProgress.put(structId, dt);
                }
            }
        }
        for (var schema : orderedInput) {
            var doc = schema.getDocument();
            for (var struct : doc.getStructs()) {
                String structId = doc.name() + "->" + struct.name();
                var toFill = structsInProgress.get(structId);
                for (String inherit : struct.getInherited()) {
                    var parent = findStructInProgress(inherit, doc);
                    toFill.inherit(parent);
                }
                for (ParsedField field : struct.getFields()) {
                    var t = resolveType(field.getType(), doc);
                    var f = new com.yahoo.document.Field(field.name(), t);
                    toFill.addField(f);
                }
            }
            for (var annotation : doc.getAnnotations()) {
                String annId = doc.name() + "->" + annotation.name();
                var at = annotationsInProgress.get(annId);
                var withStruct = annotation.getStruct();
                if (withStruct.isPresent()) {
                    ParsedStruct struct = withStruct.get();
                    String structId = doc.name() + "->" + struct.name();
                    var toFill = structsInProgress.get(structId);
                    for (ParsedField field : struct.getFields()) {
                        var t = resolveType(field.getType(), doc);
                        var f = new com.yahoo.document.Field(field.name(), t);
                        toFill.addField(f);
                    }
                    at.setDataType(toFill);
                }
                for (String inherit : annotation.getInherited()) {
                    var parent = findAnnotationInProgress(inherit, doc);
                    at.inherit(parent);
                }
            }

            var docToFill = documentsInProgress.get(doc.name());
            Map<String, Collection<String>> fieldSets = new HashMap<>();
            List<String> inDocFields = new ArrayList<>();
            for (var docField : doc.getFields()) {
                String name = docField.name();
                var t = resolveType(docField.getType(), doc);
                var f = new com.yahoo.document.Field(name, t);
                docToFill.addField(f);
                inDocFields.add(name);
            }
            fieldSets.put("[document]", inDocFields);
            for (var extraField : schema.getFields()) {
                String name = extraField.name();
                var t = resolveType(extraField.getType(), doc);
                var f = new com.yahoo.document.Field(name, t);
                docToFill.addField(f);
            }
            for (var fieldset : schema.getFieldSets()) {
                fieldSets.put(fieldset.name(), fieldset.getFieldNames());
            }
            docToFill.addFieldSets(fieldSets);
            for (String inherit : doc.getInherited()) {
                docToFill.inherit(findDocInProgress(inherit));
            }
        }
    }

    void registerDataTypes() {
        for (DataType t : structsInProgress.values()) {
            docMan.register(t);
        }
        for (DocumentType t : documentsInProgress.values()) {
            docMan.register(t);
        }
        for (AnnotationType t : annotationsInProgress.values()) {
            docMan.getAnnotationTypeRegistry().register(t);
        }
    }

    public List<Schema> convertToSchemas() {
        var resultList = new ArrayList<Schema>();
        for (var parsed : orderedInput) {
            Optional<String> inherited;
            var inheritList = parsed.getInherited();
            if (inheritList.size() == 0) {
                inherited = Optional.empty();
            } else if (inheritList.size() == 1) {
                inherited = Optional.of(inheritList.get(0));
            } else {
                throw new IllegalArgumentException("schema " + parsed.name() + "cannot inherit more than once");
            }
            Schema schema = new Schema(parsed.name(), applicationPackage, inherited, fileRegistry, deployLogger, properties);
            convertSchema(schema, parsed);
            rankProfileRegistry.add(new DefaultRankProfile(schema, rankProfileRegistry, schema.rankingConstants()));
            rankProfileRegistry.add(new UnrankedRankProfile(schema, rankProfileRegistry, schema.rankingConstants()));
            resultList.add(schema);
        }
        return resultList;
    }

    void convertMatchSettings(SDField field, ParsedMatchSettings matchInfo) {
    }

    void convertSorting(SDField field, ParsedSorting parsed, String name) {
        Attribute attribute = field.getAttributes().get(name);
        if (attribute == null) {
            attribute = new Attribute(name, field.getDataType());
            field.addAttribute(attribute);
        }
        Sorting sorting = attribute.getSorting();
        if (parsed.getAscending()) {
            sorting.setAscending();
        } else {
            sorting.setDescending();
        }
        parsed.getFunction().ifPresent(function -> sorting.setFunction(function));
        parsed.getStrength().ifPresent(strength -> sorting.setStrength(strength));
        parsed.getLocale().ifPresent(locale -> sorting.setLocale(locale));
    }

    void convertAttribute(SDField field, ParsedAttribute parsed) {
        String name = parsed.name();
        String fieldName = field.getName();
        Attribute attribute = null;
        if (fieldName.endsWith("." + name)) {
            attribute = field.getAttributes().get(field.getName());
        }
        if (attribute == null) {
            attribute = field.getAttributes().get(name);
            if (attribute == null) {
                attribute = new Attribute(name, field.getDataType());
                field.addAttribute(attribute);
            }
        }
        attribute.setHuge(parsed.getHuge());
        attribute.setPaged(parsed.getPaged());
        attribute.setFastSearch(parsed.getFastSearch());
        attribute.setFastAccess(parsed.getFastAccess());
        attribute.setMutable(parsed.getMutable());
        attribute.setEnableBitVectors(parsed.getEnableBitVectors());
        attribute.setEnableOnlyBitVector(parsed.getEnableOnlyBitVector());

        // attribute.setTensorType(?)

        for (String alias : parsed.getAliases()) {
            field.getAliasToName().put(alias, parsed.lookupAliasedFrom(alias));
        }
        var distanceMetric = parsed.getDistanceMetric();
        if (distanceMetric.isPresent()) {
            String upper = distanceMetric.get().toUpperCase(Locale.ENGLISH);
            attribute.setDistanceMetric(Attribute.DistanceMetric.valueOf(upper));
        }
        var sorting = parsed.getSorting();
        if (sorting.isPresent()) {
            convertSorting(field, sorting.get(), name);
        }
    }

    private void convertSummaryField(SDField field, ParsedSummaryField parsed, DataType type) {
        var summary = new SummaryField(parsed.name(), type);
        summary.setVsmCommand(SummaryField.VsmCommand.FLATTENSPACE); // ? always ?
        var transform = SummaryTransform.NONE;
        if (parsed.getMatchedElementsOnly()) {
            transform = SummaryTransform.MATCHED_ELEMENTS_FILTER;
        } else if (parsed.getDynamic()) {
            transform = SummaryTransform.DYNAMICTEASER;
        }
        if (parsed.getBolded()) {
            transform = transform.bold();
        }
        summary.setTransform(transform);
        if (parsed.getSources().isEmpty()) {
            summary.addSource(field.getName());
        }
        for (String source : parsed.getSources()) {
            summary.addSource(source);
        }
        if (parsed.getDestinations().isEmpty()) {
            summary.addDestination("default");
        }
        for (String destination : parsed.getDestinations()) {
            summary.addDestination(destination);
        }
        summary.setImplicit(false);
        field.addSummaryField(summary);
    }

    private void convertDocumentField(Schema schema, SDDocumentType document, ParsedField parsed, ParsedDocument owner) {
        String name = parsed.name();
        var field = new SDField(document, name, resolveType(parsed.getType(), owner));
        var indexing = parsed.getIndexing();
        if (indexing.isPresent()) {
            field.setIndexingScript(indexing.get().script());
        }
        convertMatchSettings(field, parsed.matchSettings());
        for (var dictOp : parsed.getDictionaryOptions()) {
        }
        for (var structField : field.getStructFields()) {
        }
        for (var attribute : parsed.getAttributes()) {
            convertAttribute(field, attribute);
        }
        for (var index : parsed.getIndexes()) {
        }
        for (var summaryField : parsed.getSummaryFields()) {
            var dataType = resolveType(summaryField.getType(), owner);
            convertSummaryField(field, summaryField, dataType);
        }
        for (var alias : parsed.getAliases()) {
            field.getAliasToName().put(alias, parsed.lookupAliasedFrom(alias));
        }
        for (var queryCommand : parsed.getQueryCommands()) {
        }
        for (var rankTypeEntry : parsed.getRankTypes().entrySet()) {
        }
        parsed.getSorting().ifPresent(sortInfo -> convertSorting(field, sortInfo, name));
        if (parsed.getBolding()) {
            SummaryField summaryField = field.getSummaryField(name, true);
            summaryField.addSource(name);
            summaryField.addDestination("default");
            summaryField.setTransform(summaryField.getTransform().bold());
        }
        if (parsed.getFilter()) {
            field.getRanking().setFilter(true);
        }
        document.addField(field);
    }

    private void convertDocument(Schema schema, ParsedDocument parsed) {
        SDDocumentType document = new SDDocumentType(parsed.name());
        for (String inherit : parsed.getInherited()) {
            document.inherit(new DataTypeName(inherit));
        }
        for (var struct : parsed.getStructs()) {
            String structId = parsed.name() + "->" + struct.name();
            var structProxy = new SDDocumentType(struct.name(), schema);
            structProxy.setStruct(structsInProgress.get(structId));
            document.addType(structProxy);
        }
        for (var field : parsed.getFields()) {
            convertDocumentField(schema, document, field, parsed);
        }
        schema.addDocument(document);
    }

    private void convertExtraField(Schema schema, ParsedField field) {
    }
    private void convertExtraIndex(Schema schema, ParsedIndex index) {
    }
    private void convertDocumentSummary(Schema schema, ParsedDocumentSummary docsum) {
    }
    private void convertImportField(Schema schema, ParsedSchema.ImportedField importedField) {
    }
    private void convertFieldSet(Schema schema, ParsedFieldSet fieldSet) {
    }
    private void convertRankProfile(Schema schema, ParsedRankProfile rankProfile) {
    }

    private void convertSchema(Schema schema, ParsedSchema parsed) {
        if (parsed.hasStemming()) {
            schema.setStemming(parsed.getStemming());
        }
        schema.enableRawAsBase64(parsed.getRawAsBase64());
        convertDocument(schema, parsed.getDocument());
        for (var field : parsed.getFields()) {
            convertExtraField(schema, field);
        }
        for (var index : parsed.getIndexes()) {
            convertExtraIndex(schema, index);
        }
        for (var docsum : parsed.getDocumentSummaries()) {
            convertDocumentSummary(schema, docsum);
        }
        for (var importedField : parsed.getImportedFields()) {
            convertImportField(schema, importedField);
        }
        for (var fieldSet : parsed.getFieldSets()) {
            convertFieldSet(schema, fieldSet);
        }
        for (var rankingConstant : parsed.getRankingConstants()) {
            schema.rankingConstants().add(rankingConstant);
        }
        for (var onnxModel : parsed.getOnnxModels()) {
            schema.onnxModels().add(onnxModel);
        }
        for (var rankProfile : parsed.getRankProfiles().values()) {
            convertRankProfile(schema, rankProfile);
        }
    }

}
