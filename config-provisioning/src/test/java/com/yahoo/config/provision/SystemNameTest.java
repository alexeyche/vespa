// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.config.provision;

import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.assertEquals;

/**
 * @author hakonhall
 */
public class SystemNameTest {
    @Test
    public void test() {
        for (SystemName name : SystemName.values()) {
            assertEquals(name, SystemName.from(name.value()));
        }
    }

    @Test
    public void allOf() {
        assertEquals(Set.of(SystemName.cd, SystemName.PublicCd), SystemName.allOf(SystemName::isCd));
        assertEquals(Set.of(SystemName.PublicCd, SystemName.Public), SystemName.allOf(SystemName::isPublic));
    }
}