// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.admin.task.util.fs;

import com.yahoo.vespa.hosted.node.admin.nodeagent.UserNamespace;
import com.yahoo.vespa.hosted.node.admin.nodeagent.UserScope;
import com.yahoo.vespa.hosted.node.admin.task.util.file.UnixPath;
import com.yahoo.vespa.hosted.node.admin.task.util.file.UnixUser;
import com.yahoo.vespa.test.file.TestFileSystem;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author freva
 */
class ContainerFileSystemTest {

    private final FileSystem fileSystem = TestFileSystem.create();
    private final UnixPath containerRootOnHost = new UnixPath(fileSystem.getPath("/data/storage/ctr1"));
    private final UserScope userScope = UserScope.create(new UnixUser("vespa", 1000, "users", 100), new UserNamespace(10_000, 11_000, 10000));
    private final ContainerFileSystem containerFs = ContainerFileSystem.create(containerRootOnHost.createDirectories().toPath(), userScope);

    @Test
    public void creates_files_and_directories_with_container_root_as_owner() throws IOException {
        ContainerPath containerPath = ContainerPath.fromPathInContainer(containerFs, Path.of("/opt/vespa/logs/file"), userScope.root());
        UnixPath unixPath = new UnixPath(containerPath).createParents().writeUtf8File("hello world");

        for (ContainerPath p = containerPath; p.getParent() != null; p = p.getParent())
            assertOwnership(p, 0, 0, 10000, 11000);

        unixPath.setOwnerId(500).setGroupId(1000);
        assertOwnership(containerPath, 500, 1000, 10500, 12000);

        UnixPath hostFile = new UnixPath(fileSystem.getPath("/file")).createNewFile();
        ContainerPath destination = ContainerPath.fromPathInContainer(containerFs, Path.of("/copy1"), userScope.root());
        Files.copy(hostFile.toPath(), destination);
        assertOwnership(destination, 0, 0, 10000, 11000);
    }

    @Test
    public void file_write_and_read() throws IOException {
        ContainerPath containerPath = ContainerPath.fromPathInContainer(containerFs, Path.of("/file"), userScope.root());
        UnixPath unixPath = new UnixPath(containerPath);
        unixPath.writeUtf8File("hello");
        assertOwnership(containerPath, 0, 0, 10000, 11000);

        unixPath.setOwnerId(500).setGroupId(200);
        assertOwnership(containerPath, 500, 200, 10500, 11200);
        Files.write(containerPath, " world".getBytes(StandardCharsets.UTF_8), StandardOpenOption.APPEND);
        assertOwnership(containerPath, 500, 200, 10500, 11200); // Owner should not have been updated as the file already existed

        assertEquals("hello world", unixPath.readUtf8File());

        unixPath.deleteIfExists();
        new UnixPath(containerPath.withUser(userScope.vespa())).writeUtf8File("test123");
        assertOwnership(containerPath, 1000, 100, 11000, 11100);
    }

    @Test
    public void copy() throws IOException {
        UnixPath hostFile = new UnixPath(fileSystem.getPath("/file")).createNewFile();
        ContainerPath destination = ContainerPath.fromPathInContainer(containerFs, Path.of("/dest"), userScope.root());

        // If file is copied to JimFS path, the UID/GIDs are not fixed
        Files.copy(hostFile.toPath(), destination.pathOnHost());
        assertEquals(String.valueOf(userScope.namespace().overflowId()), Files.getOwner(destination).getName());
        Files.delete(destination);

        Files.copy(hostFile.toPath(), destination);
        assertOwnership(destination, 0, 0, 10000, 11000);

        // Set owner + group on both source host file and destination container file
        hostFile.setOwnerId(5).setGroupId(10);
        new UnixPath(destination).setOwnerId(500).setGroupId(200);
        assertOwnership(destination, 500, 200, 10500, 11200);
        // Copy the host file to destination again with COPY_ATTRIBUTES and REPLACE_EXISTING
        Files.copy(hostFile.toPath(), destination, StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING);
        // The destination is recreated, so the owner should be root
        assertOwnership(destination, 0, 0, 10000, 11000);

        // Set owner + group and copy within ContainerFS
        new UnixPath(destination).setOwnerId(500).setGroupId(200);
        ContainerPath destination2 = ContainerPath.fromPathInContainer(containerFs, Path.of("/dest2"), userScope.root());
        Files.copy(destination, destination2, StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING);
        assertOwnership(destination2, 500, 200, 10500, 11200);
    }

    @Test
    public void move() throws IOException {
        UnixPath hostFile = new UnixPath(fileSystem.getPath("/file")).createNewFile();
        ContainerPath destination = ContainerPath.fromPathInContainer(containerFs, Path.of("/dest"), userScope.root());

        // If file is moved to JimFS path, the UID/GIDs are not fixed
        Files.move(hostFile.toPath(), destination.pathOnHost());
        assertEquals(String.valueOf(userScope.namespace().overflowId()), Files.getOwner(destination).getName());
        Files.delete(destination);

        hostFile.createNewFile();
        Files.move(hostFile.toPath(), destination);
        assertOwnership(destination, 0, 0, 10000, 11000);

        // Set owner + group on both source host file and destination container file
        hostFile.createNewFile();
        hostFile.setOwnerId(5).setGroupId(10);
        new UnixPath(destination).setOwnerId(500).setGroupId(200);
        assertOwnership(destination, 500, 200, 10500, 11200);
        // Move the host file to destination again with COPY_ATTRIBUTES and REPLACE_EXISTING
        Files.move(hostFile.toPath(), destination, StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING);
        // The destination is recreated, so the owner should be root
        assertOwnership(destination, 0, 0, 10000, 11000);

        // Set owner + group and move within ContainerFS
        new UnixPath(destination).setOwnerId(500).setGroupId(200);
        ContainerPath destination2 = ContainerPath.fromPathInContainer(containerFs, Path.of("/dest2"), userScope.root());
        Files.move(destination, destination2, StandardCopyOption.COPY_ATTRIBUTES, StandardCopyOption.REPLACE_EXISTING);
        assertOwnership(destination2, 500, 200, 10500, 11200);
    }

    @Test
    public void symlink() throws IOException {
        ContainerPath source = ContainerPath.fromPathInContainer(containerFs, Path.of("/src"), userScope.root());
        // Symlink from ContainerPath to some relative path (different FS provider)
        Files.createSymbolicLink(source, fileSystem.getPath("../relative/target"));
        assertEquals(fileSystem.getPath("../relative/target"), Files.readSymbolicLink(source));
        Files.delete(source);

        // Symlinks from ContainerPath to a ContainerPath: Target is resolved within container with base FS provider
        Files.createSymbolicLink(source, ContainerPath.fromPathInContainer(containerFs, Path.of("/path/in/container"), userScope.root()));
        assertEquals(fileSystem.getPath("/path/in/container"), Files.readSymbolicLink(source));
        assertOwnership(source, 0, 0, 10000, 11000);
    }

    private static void assertOwnership(ContainerPath path, int contUid, int contGid, int hostUid, int hostGid) throws IOException {
        assertOwnership(path, contUid, contGid);
        assertOwnership(path.pathOnHost(), hostUid, hostGid);
    }

    private static void assertOwnership(Path path, int uid, int gid) throws IOException {
        Map<String, Object> attrs = Files.readAttributes(path, "unix:*", LinkOption.NOFOLLOW_LINKS);
        assertEquals(uid, attrs.get("uid"));
        assertEquals(gid, attrs.get("gid"));
    }
}
