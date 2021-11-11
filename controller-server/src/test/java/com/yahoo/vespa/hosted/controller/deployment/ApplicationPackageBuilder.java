// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.controller.deployment;

import com.yahoo.component.Version;
import com.yahoo.config.application.api.ValidationId;
import com.yahoo.config.provision.AthenzDomain;
import com.yahoo.config.provision.AthenzService;
import com.yahoo.config.provision.InstanceName;
import com.yahoo.config.provision.RegionName;
import com.yahoo.security.SignatureAlgorithm;
import com.yahoo.security.X509CertificateBuilder;
import com.yahoo.security.X509CertificateUtils;
import com.yahoo.text.Text;
import com.yahoo.vespa.hosted.controller.application.pkg.ApplicationPackage;

import javax.security.auth.x500.X500Principal;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.cert.X509Certificate;
import java.text.SimpleDateFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * A builder that builds application packages for testing purposes.
 *
 * @author mpolden
 */
public class ApplicationPackageBuilder {

    private final StringBuilder prodBody = new StringBuilder();
    private final StringBuilder validationOverridesBody = new StringBuilder();
    private final StringBuilder blockChange = new StringBuilder();
    private final StringJoiner notifications = new StringJoiner("/>\n  <email ",
                                                                "<notifications>\n  <email ",
                                                                "/>\n</notifications>\n").setEmptyValue("");
    private final StringBuilder endpointsBody = new StringBuilder();
    private final StringBuilder applicationEndpointsBody = new StringBuilder();
    private final List<X509Certificate> trustedCertificates = new ArrayList<>();

    private OptionalInt majorVersion = OptionalInt.empty();
    private String instances = "default";
    private String upgradePolicy = null;
    private String upgradeRollout = null;
    private String globalServiceId = null;
    private String athenzIdentityAttributes = null;
    private String searchDefinition = "search test { }";
    private boolean explicitSystemTest = false;
    private boolean explicitStagingTest = false;
    private Version compileVersion = Version.fromString("6.1");

    public ApplicationPackageBuilder majorVersion(int majorVersion) {
        this.majorVersion = OptionalInt.of(majorVersion);
        return this;
    }

    public ApplicationPackageBuilder instances(String instances) {
        this.instances = instances;
        return this;
    }

    public ApplicationPackageBuilder upgradePolicy(String upgradePolicy) {
        this.upgradePolicy = upgradePolicy;
        return this;
    }

    public ApplicationPackageBuilder upgradeRollout(String upgradeRollout) {
        this.upgradeRollout = upgradeRollout;
        return this;
    }

    public ApplicationPackageBuilder globalServiceId(String globalServiceId) {
        this.globalServiceId = globalServiceId;
        return this;
    }

    public ApplicationPackageBuilder endpoint(String id, String containerId, String... regions) {
        endpointsBody.append("      <endpoint");
        endpointsBody.append(" id='").append(id).append("'");
        endpointsBody.append(" container-id='").append(containerId).append("'");
        endpointsBody.append(">\n");
        for (var region : regions) {
            endpointsBody.append("        <region>").append(region).append("</region>\n");
        }
        endpointsBody.append("      </endpoint>\n");
        return this;
    }

    public ApplicationPackageBuilder applicationEndpoint(String id, String containerId, String region,
                                                         Map<InstanceName, Integer> instanceWeights) {
        if (instanceWeights.isEmpty()) throw new IllegalArgumentException("At least one instance must be given");
        applicationEndpointsBody.append("    <endpoint");
        applicationEndpointsBody.append(" id='").append(id).append("'");
        applicationEndpointsBody.append(" container-id='").append(containerId).append("'");
        applicationEndpointsBody.append(" region='").append(region).append("'");
        applicationEndpointsBody.append(">\n");
        for (var kv : new TreeMap<>(instanceWeights).entrySet()) {
            applicationEndpointsBody.append("      <instance weight='").append(kv.getValue().toString()).append("'>")
                                    .append(kv.getKey().value())
                                    .append("</instance>\n");
        }
        applicationEndpointsBody.append("    </endpoint>\n");
        return this;
    }

    public ApplicationPackageBuilder systemTest() {
        explicitSystemTest = true;
        return this;
    }

    public ApplicationPackageBuilder stagingTest() {
        explicitStagingTest = true;
        return this;
    }

    public ApplicationPackageBuilder region(RegionName regionName) {
        return region(regionName, true);
    }

    public ApplicationPackageBuilder region(String regionName) {
        return region(RegionName.from(regionName), true);
    }

    public ApplicationPackageBuilder region(RegionName regionName, boolean active) {
        prodBody.append("      <region active='")
                .append(active)
                .append("'>")
                .append(regionName.value())
                .append("</region>\n");
        return this;
    }

    public ApplicationPackageBuilder test(String regionName) {
        prodBody.append("      <test>");
        prodBody.append(regionName);
        prodBody.append("</test>\n");
        return this;
    }

    public ApplicationPackageBuilder parallel(String... regionName) {
        prodBody.append("    <parallel>\n");
        Arrays.stream(regionName).forEach(this::region);
        prodBody.append("    </parallel>\n");
        return this;
    }

    public ApplicationPackageBuilder delay(Duration delay) {
        prodBody.append("    <delay seconds='");
        prodBody.append(delay.getSeconds());
        prodBody.append("'/>\n");
        return this;
    }

    public ApplicationPackageBuilder blockChange(boolean revision, boolean version, String daySpec, String hourSpec,
                                                 String zoneSpec) {
        blockChange.append("    <block-change");
        blockChange.append(" revision='").append(revision).append("'");
        blockChange.append(" version='").append(version).append("'");
        blockChange.append(" days='").append(daySpec).append("'");
        blockChange.append(" hours='").append(hourSpec).append("'");
        blockChange.append(" time-zone='").append(zoneSpec).append("'");
        blockChange.append("/>\n");
        return this;
    }

    public ApplicationPackageBuilder allow(ValidationId validationId) {
        validationOverridesBody.append("  <allow until='");
        validationOverridesBody.append(asIso8601Date(Instant.now().plus(Duration.ofDays(28))));
        validationOverridesBody.append("'>");
        validationOverridesBody.append(validationId.value());
        validationOverridesBody.append("</allow>\n");
        return this;
    }

    public ApplicationPackageBuilder compileVersion(Version version) {
        compileVersion = version;
        return this;
    }

    public ApplicationPackageBuilder athenzIdentity(AthenzDomain domain, AthenzService service) {
        this.athenzIdentityAttributes = Text.format("athenz-domain='%s' athenz-service='%s'", domain.value(),
                                                      service.value());
        return this;
    }

    public ApplicationPackageBuilder emailRole(String role) {
        this.notifications.add("role=\"" + role + "\"");
        return this;
    }

    public ApplicationPackageBuilder emailAddress(String address) {
        this.notifications.add("address=\"" + address + "\"");
        return this;
    }

    /** Sets the content of the search definition test.sd */
    public ApplicationPackageBuilder searchDefinition(String testSearchDefinition) {
        this.searchDefinition = testSearchDefinition;
        return this;
    }

    /** Add a trusted certificate to security/clients.pem */
    public ApplicationPackageBuilder trust(X509Certificate certificate) {
        this.trustedCertificates.add(certificate);
        return this;
    }

    /** Add a default trusted certificate to security/clients.pem */
    public ApplicationPackageBuilder trustDefaultCertificate() {
        try {
            var generator = KeyPairGenerator.getInstance("RSA");
            var certificate = X509CertificateBuilder.fromKeypair(
                    generator.generateKeyPair(),
                    new X500Principal("CN=name"),
                    Instant.now(),
                    Instant.now().plusMillis(300_000),
                    SignatureAlgorithm.SHA256_WITH_RSA,
                    X509CertificateBuilder.generateRandomSerialNumber()
            ).build();
            return trust(certificate);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    private byte[] deploymentSpec() {
        StringBuilder xml = new StringBuilder();
        xml.append("<deployment version='1.0' ");
        majorVersion.ifPresent(v -> xml.append("major-version='").append(v).append("' "));
        if(athenzIdentityAttributes != null) {
            xml.append(athenzIdentityAttributes);
        }
        xml.append(">\n");
        xml.append("  <instance id='").append(instances).append("'>\n");
        if (upgradePolicy != null || upgradeRollout != null) {
            xml.append("    <upgrade ");
            if (upgradePolicy != null) xml.append("policy='").append(upgradePolicy).append("' ");
            if (upgradeRollout != null) xml.append("rollout='").append(upgradeRollout).append("' ");
            xml.append("/>\n");
        }
        xml.append(notifications);
        if (explicitSystemTest)
            xml.append("    <test />\n");
        if (explicitStagingTest)
            xml.append("    <staging />\n");
        xml.append(blockChange);
        xml.append("    <prod");
        if (globalServiceId != null) {
            xml.append(" global-service-id='");
            xml.append(globalServiceId);
            xml.append("'");
        }
        xml.append(">\n");
        xml.append(prodBody);
        xml.append("    </prod>\n");
        if (endpointsBody.length() > 0 ) {
            xml.append("    <endpoints>\n");
            xml.append(endpointsBody);
            xml.append("    </endpoints>\n");
        }
        xml.append("  </instance>\n");
        if (applicationEndpointsBody.length() > 0) {
            xml.append("  <endpoints>\n");
            xml.append(applicationEndpointsBody);
            xml.append("  </endpoints>\n");
        }
        xml.append("</deployment>\n");
        return xml.toString().getBytes(UTF_8);
    }
    
    private byte[] validationOverrides() {
        String xml = "<validation-overrides version='1.0'>\n" +
                validationOverridesBody +
                "</validation-overrides>\n";
        return xml.getBytes(UTF_8);
    }

    private byte[] searchDefinition() { 
        return searchDefinition.getBytes(UTF_8);
    }

    private static byte[] buildMeta(Version compileVersion) {
        return ("{\"compileVersion\":\"" + compileVersion.toFullString() + "\",\"buildTime\":1000}").getBytes(UTF_8);
    }

    public ApplicationPackage build() {
        return build(false);
    }

    public ApplicationPackage build(boolean useApplicationDir) {
        String dir = "";
        if (useApplicationDir) {
            dir = "application/";
        }
        ByteArrayOutputStream zip = new ByteArrayOutputStream();
        try (ZipOutputStream out = new ZipOutputStream(zip)) {
            out.setLevel(Deflater.NO_COMPRESSION); // This is for testing purposes so we skip compression for performance
            writeZipEntry(out, dir + "deployment.xml", deploymentSpec());
            writeZipEntry(out, dir + "validation-overrides.xml", validationOverrides());
            writeZipEntry(out, dir + "search-definitions/test.sd", searchDefinition());
            writeZipEntry(out, dir + "build-meta.json", buildMeta(compileVersion));
            if (!trustedCertificates.isEmpty()) {
                writeZipEntry(out, dir + "security/clients.pem", X509CertificateUtils.toPem(trustedCertificates).getBytes(UTF_8));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return new ApplicationPackage(zip.toByteArray());
    }

    private void writeZipEntry(ZipOutputStream out, String name, byte[] content) throws IOException {
        ZipEntry entry = new ZipEntry(name);
        out.putNextEntry(entry);
        out.write(content);
        out.closeEntry();
    }

    private static String asIso8601Date(Instant instant) {
        return new SimpleDateFormat("yyyy-MM-dd").format(Date.from(instant));
    }

    public static ApplicationPackage fromDeploymentXml(String deploymentXml) {
        ByteArrayOutputStream zip = new ByteArrayOutputStream();
        try (ZipOutputStream out = new ZipOutputStream(zip)) {
            out.putNextEntry(new ZipEntry("deployment.xml"));
            out.write(deploymentXml.getBytes(UTF_8));
            out.closeEntry();
            out.putNextEntry(new ZipEntry("build-meta.json"));
            out.write(buildMeta(Version.fromString("6.1")));
            out.closeEntry();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return new ApplicationPackage(zip.toByteArray());
    }

}
