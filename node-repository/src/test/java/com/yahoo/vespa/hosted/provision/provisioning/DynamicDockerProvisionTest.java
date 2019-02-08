// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.provision.provisioning;

import com.google.common.collect.ImmutableSet;
import com.yahoo.component.Version;
import com.yahoo.config.provision.ApplicationId;
import com.yahoo.config.provision.Capacity;
import com.yahoo.config.provision.ClusterSpec;
import com.yahoo.config.provision.Flavor;
import com.yahoo.config.provision.HostSpec;
import com.yahoo.config.provision.NodeType;
import com.yahoo.vespa.flags.Flags;
import com.yahoo.vespa.flags.InMemoryFlagSource;
import com.yahoo.vespa.hosted.provision.Node;
import com.yahoo.vespa.hosted.provision.node.Agent;
import com.yahoo.vespa.hosted.provision.testutils.MockNameResolver;
import org.junit.Test;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

/**
 * @author freva
 */
public class DynamicDockerProvisionTest {

    private final MockNameResolver nameResolver = new MockNameResolver().mockAnyLookup();
    private final HostProvisioner hostProvisioner = mock(HostProvisioner.class);
    private final InMemoryFlagSource flagSource = new InMemoryFlagSource()
            .withBooleanFlag(Flags.ENABLE_DYNAMIC_PROVISIONING.id(), true);
    private final ProvisioningTester tester = new ProvisioningTester.Builder()
            .hostProvisioner(hostProvisioner).flagSource(flagSource).nameResolver(nameResolver).build();

    @Test
    public void dynamically_provision_with_empty_node_repo() {
        assertEquals(0, tester.nodeRepository().list().size());

        ApplicationId application1 = tester.makeApplicationId();
        Flavor flavor = tester.nodeRepository().getAvailableFlavors().getFlavorOrThrow("dockerSmall");

        mockHostProvisioner(hostProvisioner, tester.nodeRepository().getAvailableFlavors().getFlavorOrThrow("small"));
        List<HostSpec> hostSpec = tester.prepare(application1, clusterSpec("myContent.t1.a1"), 4, 1, flavor.canonicalName());
        verify(hostProvisioner).provisionHosts(4, flavor);

        // Total of 8 nodes should now be in node-repo, 4 hosts in state provisioned, and 4 reserved nodes
        assertEquals(8, tester.nodeRepository().list().size());
        assertEquals(4, tester.nodeRepository().getNodes(NodeType.host, Node.State.provisioned).size());
        assertEquals(4, tester.nodeRepository().getNodes(NodeType.tenant, Node.State.reserved).size());
        assertEquals(List.of("host-1-1", "host-2-1", "host-3-1", "host-4-1"),
                hostSpec.stream().map(HostSpec::hostname).collect(Collectors.toList()));
    }

    @Test
    public void does_not_allocate_to_available_empty_hosts() {
        tester.makeReadyNodes(3, "small", NodeType.host, 10);
        deployZoneApp(tester);

        ApplicationId application = tester.makeApplicationId();
        Flavor flavor = tester.nodeRepository().getAvailableFlavors().getFlavorOrThrow("dockerSmall");

        mockHostProvisioner(hostProvisioner, tester.nodeRepository().getAvailableFlavors().getFlavorOrThrow("small"));
        tester.prepare(application, clusterSpec("myContent.t2.a2"), 2, 1, flavor.canonicalName());
        verify(hostProvisioner).provisionHosts(2, flavor);
    }

    @Test
    public void allocates_to_hosts_already_hosting_nodes_by_this_tenant() {
        ApplicationId application = tester.makeApplicationId();
        Flavor flavor = tester.nodeRepository().getAvailableFlavors().getFlavorOrThrow("dockerSmall");

        mockHostProvisioner(hostProvisioner, tester.nodeRepository().getAvailableFlavors().getFlavorOrThrow("large"));
        tester.prepare(application, clusterSpec("myContent.t2.a2"), 2, 1, flavor.canonicalName());
        verify(hostProvisioner).provisionHosts(2, flavor);

        // Ready the provisioned hosts, add an IP addreses to pool and activate them
        for (int i = 1; i < 3; i++) {
            String hostname = "host-" + i;
            Node host = tester.nodeRepository().getNode(hostname).orElseThrow()
                    .withIpAddressPool(Set.of("::" + i + ":2")).withIpAddresses(Set.of("::" + i + ":0"));
            tester.nodeRepository().setReady(List.of(host), Agent.system, getClass().getSimpleName());
            nameResolver.addRecord(hostname + "-2", "::" + i + ":2");
        }
        deployZoneApp(tester);

        mockHostProvisioner(hostProvisioner, tester.nodeRepository().getAvailableFlavors().getFlavorOrThrow("small"));
        tester.prepare(application, clusterSpec("another-id"), 2, 1, flavor.canonicalName());
        // Verify there was only 1 call to provision hosts (during the first prepare)
        verify(hostProvisioner).provisionHosts(anyInt(), any());

        // Node-repo should now consist of 2 active hosts with 2 reserved nodes on each
        assertEquals(6, tester.nodeRepository().list().size());
        assertEquals(2, tester.nodeRepository().getNodes(NodeType.host, Node.State.active).size());
        assertEquals(4, tester.nodeRepository().getNodes(NodeType.tenant, Node.State.reserved).size());
    }


    private static void deployZoneApp(ProvisioningTester tester) {
        ApplicationId applicationId = tester.makeApplicationId();
        List<HostSpec> list = tester.prepare(applicationId,
                ClusterSpec.request(ClusterSpec.Type.container,
                        ClusterSpec.Id.from("node-admin"),
                        Version.fromString("6.42"),
                        false, Collections.emptySet()),
                Capacity.fromRequiredNodeType(NodeType.host),
                1);
        tester.activate(applicationId, ImmutableSet.copyOf(list));
    }


    private static ClusterSpec clusterSpec(String clusterId) {
        return ClusterSpec.request(ClusterSpec.Type.content, ClusterSpec.Id.from(clusterId), Version.fromString("6.42"), false, Collections.emptySet());
    }

    private static void mockHostProvisioner(HostProvisioner hostProvisioner, Flavor hostFlavor) {
        final int[] numProvisioned = { 0 };
        doAnswer(invocation -> {
            int numHosts = (int) invocation.getArguments()[0];
            Flavor nodeFlavor = (Flavor) invocation.getArguments()[1];
            return IntStream.range(0, numHosts)
                    .map(i -> ++numProvisioned[0])
                    .mapToObj(i -> new ProvisionedHost("id-" + i, "host-" + i, hostFlavor, "host-" + i + "-1", nodeFlavor))
                    .collect(Collectors.toList());
        }).when(hostProvisioner).provisionHosts(anyInt(), any());
    }
}
