// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.searchdefinition;

import com.yahoo.config.application.api.ApplicationPackage;
import com.yahoo.config.application.api.DeployLogger;
import com.yahoo.config.application.api.FileRegistry;
import com.yahoo.config.model.api.ModelContext;
import com.yahoo.config.model.application.provider.BaseDeployLogger;
import com.yahoo.config.model.application.provider.MockFileRegistry;
import com.yahoo.config.model.deploy.TestProperties;
import com.yahoo.config.model.test.MockApplicationPackage;
import com.yahoo.document.DocumentTypeManager;
import com.yahoo.io.IOUtils;
import com.yahoo.io.reader.NamedReader;
import com.yahoo.path.Path;
import com.yahoo.search.query.profile.QueryProfileRegistry;
import com.yahoo.search.query.profile.config.QueryProfileXMLReader;
import com.yahoo.searchdefinition.parser.IntermediateParser;
import com.yahoo.searchdefinition.parser.IntermediateCollection;
import com.yahoo.searchdefinition.parser.ConvertSchemaCollection;
import com.yahoo.searchdefinition.parser.ParseException;
import com.yahoo.searchdefinition.parser.SimpleCharStream;
import com.yahoo.searchdefinition.parser.TokenMgrException;
import com.yahoo.searchdefinition.processing.Processor;
import com.yahoo.vespa.documentmodel.DocumentModel;
import com.yahoo.vespa.model.container.search.QueryProfiles;
import com.yahoo.yolean.Exceptions;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Application builder. Usage:
 * 1) Add all schemas, using the addXXX() methods,
 * 3) invoke the {@link #build} method
 */
public class NewApplicationBuilder {

    private final IntermediateCollection mediator;
    private final ApplicationPackage applicationPackage;
    private final DocumentTypeManager documentTypeManager = new DocumentTypeManager();
    private final RankProfileRegistry rankProfileRegistry;
    private final QueryProfileRegistry queryProfileRegistry;
    private final FileRegistry fileRegistry;
    private final DeployLogger deployLogger;
    private final ModelContext.Properties properties;
    /** True to build the document aspect only, skipping instantiation of rank profiles */
    private final boolean documentsOnly;

    private Application application;

    private final Set<Class<? extends Processor>> processorsToSkip = new HashSet<>();

    /** For testing only */
    public NewApplicationBuilder() {
        this(new RankProfileRegistry(), new QueryProfileRegistry());
    }

    /** For testing only */
    public NewApplicationBuilder(DeployLogger deployLogger) {
        this(MockApplicationPackage.createEmpty(), deployLogger);
    }

    /** For testing only */
    public NewApplicationBuilder(DeployLogger deployLogger, RankProfileRegistry rankProfileRegistry) {
        this(MockApplicationPackage.createEmpty(), deployLogger, rankProfileRegistry);
    }

    /** Used for generating documents for typed access to document fields in Java */
    public NewApplicationBuilder(boolean documentsOnly) {
        this(MockApplicationPackage.createEmpty(), new MockFileRegistry(), new BaseDeployLogger(), new TestProperties(), new RankProfileRegistry(), new QueryProfileRegistry(), documentsOnly);
    }

    /** For testing only */
    public NewApplicationBuilder(ApplicationPackage app, DeployLogger deployLogger) {
        this(app, new MockFileRegistry(), deployLogger, new TestProperties(), new RankProfileRegistry(), new QueryProfileRegistry());
    }

    /** For testing only */
    public NewApplicationBuilder(ApplicationPackage app, DeployLogger deployLogger, RankProfileRegistry rankProfileRegistry) {
        this(app, new MockFileRegistry(), deployLogger, new TestProperties(), rankProfileRegistry, new QueryProfileRegistry());
    }

    /** For testing only */
    public NewApplicationBuilder(RankProfileRegistry rankProfileRegistry) {
        this(rankProfileRegistry, new QueryProfileRegistry());
    }

    /** For testing only */
    public NewApplicationBuilder(RankProfileRegistry rankProfileRegistry, QueryProfileRegistry queryProfileRegistry) {
        this(rankProfileRegistry, queryProfileRegistry, new TestProperties());
    }

    public NewApplicationBuilder(RankProfileRegistry rankProfileRegistry, QueryProfileRegistry queryProfileRegistry, ModelContext.Properties properties) {
        this(MockApplicationPackage.createEmpty(), new MockFileRegistry(), new BaseDeployLogger(), properties, rankProfileRegistry, queryProfileRegistry);
    }

    public NewApplicationBuilder(ApplicationPackage app,
                                 FileRegistry fileRegistry,
                                 DeployLogger deployLogger,
                                 ModelContext.Properties properties,
                                 RankProfileRegistry rankProfileRegistry,
                                 QueryProfileRegistry queryProfileRegistry) {
        this(app, fileRegistry, deployLogger, properties, rankProfileRegistry, queryProfileRegistry, false);
    }

    private NewApplicationBuilder(ApplicationPackage applicationPackage,
                                  FileRegistry fileRegistry,
                                  DeployLogger deployLogger,
                                  ModelContext.Properties properties,
                                  RankProfileRegistry rankProfileRegistry,
                                  QueryProfileRegistry queryProfileRegistry,
                                  boolean documentsOnly) {
        this.applicationPackage = applicationPackage;
        this.rankProfileRegistry = rankProfileRegistry;
        this.queryProfileRegistry = queryProfileRegistry;
        this.fileRegistry = fileRegistry;
        this.deployLogger = deployLogger;
        this.properties = properties;
        this.documentsOnly = documentsOnly;
        this.mediator = new IntermediateCollection(deployLogger, properties);
        for (NamedReader reader : applicationPackage.getSchemas())
            addSchema(reader);
    }

    /**
     * Adds a schema to this application.
     *
     * @param fileName the name of the file to import
     * @return the name of the imported object
     * @throws IOException    thrown if the file can not be read for some reason
     * @throws ParseException thrown if the file does not contain a valid search definition
     */
    public void addSchemaFile(String fileName) throws IOException, ParseException {
        mediator.addSchemaFromFile(fileName);
    }

    /**
     * Reads and parses the schema string provided by the given reader. Once all schemas have been
     * imported, call {@link #build}.
     *
     * @param reader the reader whose content to import
     */
    public void addSchema(NamedReader reader) {
        var parsedName = mediator.addSchemaFromReader(reader);
        addRankProfileFiles(parsedName);
    }

    /**
     * Adds a schema to this
     *
     * @param schemaString the content of the schema
     */
    public void addSchema(String schemaString) throws ParseException {
        mediator.addSchemaFromString(schemaString);
    }

    private void addRankProfileFiles(String schemaName) {
        if (applicationPackage == null) return;

        Path legacyRankProfilePath = ApplicationPackage.SEARCH_DEFINITIONS_DIR.append(schemaName);
        for (NamedReader reader : applicationPackage.getFiles(legacyRankProfilePath, ".profile"))
            mediator.addRankProfileFile(schemaName, reader);

        Path rankProfilePath = ApplicationPackage.SCHEMAS_DIR.append(schemaName);
        for (NamedReader reader : applicationPackage.getFiles(rankProfilePath, ".profile"))
            mediator.addRankProfileFile(schemaName, reader);
    }

    /**
     * Processes and finalizes the schemas of this.
     *
     * @throws IllegalStateException thrown if this method has already been called
     */
    public Application build(boolean validate) {
        var converter = new ConvertSchemaCollection(mediator,
                                                    documentTypeManager,
                                                    applicationPackage,
                                                    fileRegistry,
                                                    deployLogger,
                                                    properties,
                                                    rankProfileRegistry,
                                                    documentsOnly);
        List<Schema> schemas = converter.convertToSchemas();
        application = new Application(applicationPackage,
                                      schemas,
                                      rankProfileRegistry,
                                      new QueryProfiles(queryProfileRegistry, deployLogger),
                                      properties,
                                      documentsOnly,
                                      validate,
                                      processorsToSkip,
                                      deployLogger);
        return application;
    }

    /** Returns a modifiable set of processors we should skip for these schemas. Useful for testing. */
    public Set<Class<? extends Processor>> processorsToSkip() { return processorsToSkip; }

    /**
     * Convenience method to call {@link #getSchema(String)} when there is only a single {@link Schema} object
     * built. This method will never return null.
     *
     * @return the built object
     * @throws IllegalStateException if there is not exactly one search.
     */
    public Schema getSchema() {
        if (application == null) {
            throw new IllegalStateException("Application not built");
        }
        if (application.schemas().size() != 1) {
            throw new IllegalStateException("This call only works if we have 1 schema. Schemas: " +
                                            application.schemas().values());
        }
        return application.schemas().values().stream().findAny().get();
    }

    public DocumentModel getModel() { return application.documentModel(); }

    /**
     * Returns the built {@link Schema} object that has the given name. If the name is unknown, this method will simply
     * return null.
     *
     * @param name the name of the schema to return,
     *             or null to return the only one or throw an exception if there are multiple to choose from
     * @return the built object, or null if none with this name
     * @throws IllegalStateException if {@link #build} has not been called.
     */
    public Schema getSchema(String name) {
        if (application == null)  throw new IllegalStateException("Application not built");
        if (name == null) return getSchema();
        return application.schemas().get(name);
    }

    public Application application() { return application; }

    /**
     * Convenience method to return a list of all built {@link Schema} objects.
     *
     * @return the list of built searches
     */
    public List<Schema> getSchemaList() {
        return new ArrayList<>(application.schemas().values());
    }

    /**
     * Convenience factory method to import and build a {@link Schema} object from a string.
     *
     * @param sd the string to build from
     * @return the built {@link ApplicationBuilder} object
     * @throws ParseException thrown if there is a problem parsing the string
     */
    public static NewApplicationBuilder createFromString(String sd) throws ParseException {
        return createFromString(sd, new BaseDeployLogger());
    }

    public static NewApplicationBuilder createFromString(String sd, DeployLogger logger) throws ParseException {
        NewApplicationBuilder builder = new NewApplicationBuilder(logger);
        builder.addSchema(sd);
        builder.build(true);
        return builder;
    }

    public static NewApplicationBuilder createFromStrings(DeployLogger logger, String ... schemas) throws ParseException {
        NewApplicationBuilder builder = new NewApplicationBuilder(logger);
        for (var schema : schemas)
            builder.addSchema(schema);
        builder.build(true);
        return builder;
    }

    /**
     * Convenience factory method to import and build a {@link Schema} object from a file. Only for testing.
     *
     * @param fileName the file to build from
     * @return the built {@link NewApplicationBuilder} object
     * @throws IOException    if there was a problem reading the file.
     * @throws ParseException if there was a problem parsing the file content.
     */
    public static NewApplicationBuilder createFromFile(String fileName) throws IOException, ParseException {
        return createFromFile(fileName, new BaseDeployLogger());
    }

    /**
     * Convenience factory methdd to create a SearchBuilder from multiple SD files. Only for testing.
     */
    public static NewApplicationBuilder createFromFiles(Collection<String> fileNames) throws IOException, ParseException {
        return createFromFiles(fileNames, new BaseDeployLogger());
    }

    public static NewApplicationBuilder createFromFile(String fileName, DeployLogger logger) throws IOException, ParseException {
        return createFromFile(fileName, logger, new RankProfileRegistry(), new QueryProfileRegistry());
    }

    private static NewApplicationBuilder createFromFiles(Collection<String> fileNames, DeployLogger logger) throws IOException, ParseException {
        return createFromFiles(fileNames, new MockFileRegistry(), logger, new TestProperties(), new RankProfileRegistry(), new QueryProfileRegistry());
    }

    /**
     * Convenience factory method to import and build a {@link Schema} object from a file.
     *
     * @param fileName the file to build from.
     * @param deployLogger logger for deploy messages.
     * @param rankProfileRegistry registry for rank profiles.
     * @return the built {@link NewApplicationBuilder} object.
     * @throws IOException    if there was a problem reading the file.
     * @throws ParseException if there was a problem parsing the file content.
     */
    private static NewApplicationBuilder createFromFile(String fileName,
                                                        DeployLogger deployLogger,
                                                        RankProfileRegistry rankProfileRegistry,
                                                        QueryProfileRegistry queryprofileRegistry)
        throws IOException, ParseException {
        return createFromFiles(Collections.singletonList(fileName), new MockFileRegistry(), deployLogger, new TestProperties(),
                               rankProfileRegistry, queryprofileRegistry);
    }

    /**
     * Convenience factory methdd to create a SearchBuilder from multiple SD files..
     */
    private static NewApplicationBuilder createFromFiles(Collection<String> fileNames,
                                                         FileRegistry fileRegistry,
                                                         DeployLogger deployLogger,
                                                         ModelContext.Properties properties,
                                                         RankProfileRegistry rankProfileRegistry,
                                                         QueryProfileRegistry queryprofileRegistry)
        throws IOException, ParseException {
        NewApplicationBuilder builder = new NewApplicationBuilder(MockApplicationPackage.createEmpty(),
                                                                  fileRegistry,
                                                                  deployLogger,
                                                                  properties,
                                                                  rankProfileRegistry,
                                                                  queryprofileRegistry);
        for (String fileName : fileNames) {
            builder.addSchemaFile(fileName);
        }
        builder.build(true);
        return builder;
    }


    public static NewApplicationBuilder createFromDirectory(String dir, FileRegistry fileRegistry, DeployLogger logger, ModelContext.Properties properties) throws IOException, ParseException {
        return createFromDirectory(dir, fileRegistry, logger, properties, new RankProfileRegistry());
    }
    public static NewApplicationBuilder createFromDirectory(String dir,
                                                            FileRegistry fileRegistry,
                                                            DeployLogger logger,
                                                            ModelContext.Properties properties,
                                                            RankProfileRegistry rankProfileRegistry) throws IOException, ParseException {
        return createFromDirectory(dir, fileRegistry, logger, properties, rankProfileRegistry, createQueryProfileRegistryFromDirectory(dir));
    }
    private static NewApplicationBuilder createFromDirectory(String dir,
                                                          FileRegistry fileRegistry,
                                                          DeployLogger logger,
                                                          ModelContext.Properties properties,
                                                          RankProfileRegistry rankProfileRegistry,
                                                          QueryProfileRegistry queryProfileRegistry) throws IOException, ParseException {
        return createFromDirectory(dir, MockApplicationPackage.fromSearchDefinitionAndRootDirectory(dir), fileRegistry, logger, properties,
                                   rankProfileRegistry, queryProfileRegistry);
    }

    private static NewApplicationBuilder createFromDirectory(String dir,
                                                          ApplicationPackage applicationPackage,
                                                          FileRegistry fileRegistry,
                                                          DeployLogger deployLogger,
                                                          ModelContext.Properties properties,
                                                          RankProfileRegistry rankProfileRegistry,
                                                          QueryProfileRegistry queryProfileRegistry) throws IOException, ParseException {
        NewApplicationBuilder builder = new NewApplicationBuilder(applicationPackage,
                                                            fileRegistry,
                                                            deployLogger,
                                                            properties,
                                                            rankProfileRegistry,
                                                            queryProfileRegistry);
        for (var i = Files.list(new File(dir).toPath()).filter(p -> p.getFileName().toString().endsWith(".sd")).iterator(); i.hasNext(); ) {
            builder.addSchemaFile(i.next().toString());
        }
        builder.build(true);
        return builder;
    }

    private static QueryProfileRegistry createQueryProfileRegistryFromDirectory(String dir) {
        File queryProfilesDir = new File(dir, "query-profiles");
        if ( ! queryProfilesDir.exists()) return new QueryProfileRegistry();
        return new QueryProfileXMLReader().read(queryProfilesDir.toString());
    }

    public RankProfileRegistry getRankProfileRegistry() {
        return rankProfileRegistry;
    }

    public QueryProfileRegistry getQueryProfileRegistry() {
        return queryProfileRegistry;
    }

    public ModelContext.Properties getProperties() { return properties; }

    public DeployLogger getDeployLogger() { return deployLogger; }

}
