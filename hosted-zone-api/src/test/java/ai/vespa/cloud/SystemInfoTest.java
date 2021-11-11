// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.cloud;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * @author bratseth
 */
public class SystemInfoTest {

    @Test
    public void testSystemInfo() {
        ApplicationId application = new ApplicationId("tenant1", "application1", "instance1");
        Zone zone = new Zone(Environment.dev, "us-west-1");
        Cluster cluster = new Cluster(1, List.of());
        Node node = new Node(0);

        SystemInfo info = new SystemInfo(application, zone, cluster, node);
        assertEquals(application, info.application());
        assertEquals(zone, info.zone());
        assertEquals(cluster, info.cluster());
        assertEquals(node, info.node());
    }

    @Test
    public void testZone() {
        Zone zone = Zone.from("dev.us-west-1");
        zone = Zone.from(zone.toString());
        assertEquals(Environment.dev, zone.environment());
        assertEquals("us-west-1", zone.region());
        Zone sameZone = Zone.from("dev.us-west-1");
        assertEquals(sameZone.hashCode(), zone.hashCode());
        assertEquals(sameZone, zone);

        try {
            Zone.from("invalid");
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("A zone string must be on the form [environment].[region], but was 'invalid'",
                         e.getMessage());
        }

        try {
            Zone.from("invalid.us-west-1");
            fail("Expected exception");
        }
        catch (IllegalArgumentException e) {
            assertEquals("Invalid zone 'invalid.us-west-1': No environment named 'invalid'", e.getMessage());
        }
    }

    @Test
    public void testCluster() {
        int size = 1;
        var indices = List.of(1);
        Cluster cluster = new Cluster(size, indices);
        assertEquals(size, cluster.size());
        assertEquals(indices, cluster.indices());
    }

    @Test
    public void testNode() {
        int index = 0;
        Node node = new Node(index);
        assertEquals(index, node.index());
    }

}
