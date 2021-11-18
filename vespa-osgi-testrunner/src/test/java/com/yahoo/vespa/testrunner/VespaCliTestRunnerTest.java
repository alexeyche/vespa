// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.testrunner;

import ai.vespa.hosted.api.TestConfig;
import com.yahoo.config.provision.ApplicationId;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author jonmv
 */
class VespaCliTestRunnerTest {

    static final TestConfig testConfig = TestConfig.fromJson(("{\n" +
                                                              "  \"application\": \"t:a:i\",\n" +
                                                              "  \"zone\": \"dev.aws-us-east-1c\",\n" +
                                                              "  \"system\": \"publiccd\",\n" +
                                                              "  \"isCI\": true,\n" +
                                                              "  \"zoneEndpoints\": {\n" +
                                                              "    \"dev.aws-us-east-1c\": {\n" +
                                                              "      \"default\": \"https://dev.endpoint:443/\"\n" +
                                                              "    },\n" +
                                                              "    \"prod.aws-us-east-1a\": {\n" +
                                                              "      \"default\": \"https://prod.endpoint:443/\"\n" +
                                                              "    }\n" +
                                                              "  },\n" +
                                                              "  \"clusters\": {\n" +
                                                              "    \"prod.aws-us-east-1c\": [\n" +
                                                              "      \"documents\"\n" +
                                                              "    ]\n" +
                                                              "  }\n" +
                                                              "}\n").getBytes(StandardCharsets.UTF_8));

    @Test
    void testSetup() throws IOException {
        Path temp = Files.createTempDirectory("vespa-cli-test-runner-test-");
        temp.toFile().deleteOnExit();
        VespaCliTestRunner runner = new VespaCliTestRunner(temp);
        assertFalse(runner.isSupported());

        Path tests = Files.createDirectory(temp.resolve("tests"));
        assertTrue(runner.isSupported());
        IllegalStateException ise = assertThrows(IllegalStateException.class,
                                                 () -> runner.testRunProcessBuilder(TestRunner.Suite.SYSTEM_TEST, testConfig));
        assertEquals("No tests found, for suite 'SYSTEM_TEST'", ise.getMessage());

        Path systemTests = Files.createDirectory(tests.resolve("system-test"));
        ProcessBuilder builder = runner.testRunProcessBuilder(TestRunner.Suite.SYSTEM_TEST, testConfig);
        assertEquals(systemTests.toFile(), builder.directory());
        assertEquals(List.of("vespa", "test",
                             "--application", "t.a.i",
                             "--endpoints", "{\"endpoints\":[{\"cluster\":\"default\",\"url\":\"https://dev.endpoint:443/\"}]}"),
                     builder.command());

        Path credentialsPath = temp.resolve("creds");
        assertThrows(NoSuchFileException.class,
                     () -> runner.copyCredentials(credentialsPath));

        Files.write(temp.resolve("key"), new byte[]{ 0 });
        Files.write(temp.resolve("cert"), new byte[]{ 1 });
        runner.copyCredentials(credentialsPath);
        assertArrayEquals(new byte[]{ 0 }, Files.readAllBytes(credentialsPath.resolve("data-plane-private-key.pem")));
        assertArrayEquals(new byte[]{ 1 }, Files.readAllBytes(credentialsPath.resolve("data-plane-public-cert.pem")));
    }

}
