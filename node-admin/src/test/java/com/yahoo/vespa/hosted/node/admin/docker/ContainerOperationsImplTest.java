// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.admin.docker;

import com.google.common.net.InetAddresses;
import com.yahoo.config.provision.DockerImage;
import com.yahoo.vespa.hosted.dockerapi.Container;
import com.yahoo.vespa.hosted.dockerapi.ContainerEngine;
import com.yahoo.vespa.hosted.dockerapi.ContainerName;
import com.yahoo.vespa.hosted.dockerapi.ProcessResult;
import com.yahoo.vespa.hosted.node.admin.nodeagent.ContainerData;
import com.yahoo.vespa.hosted.node.admin.nodeagent.NodeAgentContext;
import com.yahoo.vespa.hosted.node.admin.nodeagent.NodeAgentContextImpl;
import com.yahoo.vespa.hosted.node.admin.task.util.network.IPAddresses;
import com.yahoo.vespa.hosted.node.admin.task.util.network.IPAddressesMock;
import com.yahoo.vespa.hosted.node.admin.task.util.process.TestTerminal;
import com.yahoo.vespa.test.file.TestFileSystem;
import org.junit.Test;
import org.mockito.InOrder;

import java.net.InetAddress;
import java.nio.file.FileSystem;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ContainerOperationsImplTest {
    private final ContainerEngine containerEngine = mock(ContainerEngine.class);
    private final TestTerminal terminal = new TestTerminal();
    private final IPAddresses ipAddresses = new IPAddressesMock();
    private final FileSystem fileSystem = TestFileSystem.create();
    private final ContainerOperationsImpl containerOperations = new ContainerOperationsImpl(
            containerEngine, terminal, ipAddresses, fileSystem);

    @Test
    public void processResultFromNodeProgramWhenSuccess() {
        final NodeAgentContext context = new NodeAgentContextImpl.Builder("container-123.domain.tld").build();
        final ProcessResult actualResult = new ProcessResult(0, "output", "errors");

        when(containerEngine.executeInContainerAsUser(any(), any(), any(), any()))
                .thenReturn(actualResult); // output from node program

        ProcessResult result = containerOperations.executeNodeCtlInContainer(context, "start");

        final InOrder inOrder = inOrder(containerEngine);
        inOrder.verify(containerEngine, times(1)).executeInContainerAsUser(
                eq(context.containerName()),
                eq("root"),
                eq(OptionalLong.empty()),
                eq("/opt/vespa/bin/vespa-nodectl"),
                eq("start"));

        assertThat(result, is(actualResult));
    }

    @Test(expected = RuntimeException.class)
    public void processResultFromNodeProgramWhenNonZeroExitCode() {
        final NodeAgentContext context = new NodeAgentContextImpl.Builder("container-123.domain.tld").build();
        final ProcessResult actualResult = new ProcessResult(3, "output", "errors");

        when(containerEngine.executeInContainerAsUser(any(), any(), any(), any()))
                .thenReturn(actualResult); // output from node program

        containerOperations.executeNodeCtlInContainer(context, "start");
    }

    @Test
    public void runsCommandInNetworkNamespace() {
        NodeAgentContext context = new NodeAgentContextImpl.Builder("container-42.domain.tld").build();
        makeContainer("container-42", Container.State.RUNNING, 42);

        terminal.expectCommand("nsenter --net=/proc/42/ns/net -- iptables -nvL 2>&1");

        containerOperations.executeCommandInNetworkNamespace(context, "iptables", "-nvL");
    }

    private Container makeContainer(String name, Container.State state, int pid) {
        final Container container = new Container(name + ".fqdn", DockerImage.fromString("registry.example.com/mock"), null,
                new ContainerName(name), state, pid);
        when(containerEngine.getContainer(eq(container.name))).thenReturn(Optional.of(container));
        return container;
    }

    @Test
    public void verifyEtcHosts() {
        ContainerData containerData = mock(ContainerData.class);
        String hostname = "hostname";
        InetAddress ipV6Local = InetAddresses.forString("::1");
        InetAddress ipV4Local = InetAddresses.forString("127.0.0.1");

        containerOperations.addEtcHosts(containerData, hostname, Optional.empty(), Optional.of(ipV6Local));

        verify(containerData, times(1)).addFile(
                fileSystem.getPath("/etc/hosts"),
                "# This file was generated by com.yahoo.vespa.hosted.node.admin.docker.ContainerOperationsImpl\n" +
                        "127.0.0.1	localhost\n" +
                        "::1	localhost ip6-localhost ip6-loopback\n" +
                        "fe00::0	ip6-localnet\n" +
                        "ff00::0	ip6-mcastprefix\n" +
                        "ff02::1	ip6-allnodes\n" +
                        "ff02::2	ip6-allrouters\n" +
                        "0:0:0:0:0:0:0:1	hostname\n");

        containerOperations.addEtcHosts(containerData, hostname, Optional.of(ipV4Local), Optional.of(ipV6Local));

        verify(containerData, times(1)).addFile(
                fileSystem.getPath("/etc/hosts"),
                "# This file was generated by com.yahoo.vespa.hosted.node.admin.docker.ContainerOperationsImpl\n" +
                        "127.0.0.1	localhost\n" +
                        "::1	localhost ip6-localhost ip6-loopback\n" +
                        "fe00::0	ip6-localnet\n" +
                        "ff00::0	ip6-mcastprefix\n" +
                        "ff02::1	ip6-allnodes\n" +
                        "ff02::2	ip6-allrouters\n" +
                        "0:0:0:0:0:0:0:1	hostname\n" +
                        "127.0.0.1	hostname\n");
    }

    @Test
    public void retainContainersTest() {
        when(containerEngine.listManagedContainers(ContainerOperationsImpl.MANAGER_NAME))
                .thenReturn(List.of(new ContainerName("cnt1"), new ContainerName("cnt2"), new ContainerName("cnt3")));
        containerOperations.retainManagedContainers(Set.of(new ContainerName("cnt2"), new ContainerName("cnt4")));

        verify(containerEngine).stopContainer(eq(new ContainerName("cnt1")));
        verify(containerEngine).deleteContainer(eq(new ContainerName("cnt1")));
        verify(containerEngine).stopContainer(eq(new ContainerName("cnt3")));
        verify(containerEngine).deleteContainer(eq(new ContainerName("cnt3")));
    }
}
