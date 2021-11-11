// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.config.server.maintenance;

import com.yahoo.vespa.config.server.ApplicationRepository;
import com.yahoo.vespa.curator.Curator;
import com.yahoo.vespa.flags.FlagSource;

import java.time.Duration;
import java.util.logging.Level;

/**
 * Removes expired sessions
 * <p>
 * Note: Unit test is in ApplicationRepositoryTest
 *
 * @author hmusum
 */
public class SessionsMaintainer extends ConfigServerMaintainer {
    private final boolean hostedVespa;

    SessionsMaintainer(ApplicationRepository applicationRepository, Curator curator, Duration interval, FlagSource flagSource) {
        super(applicationRepository, curator, flagSource, applicationRepository.clock().instant(), interval, true);
        this.hostedVespa = applicationRepository.configserverConfig().hostedVespa();
    }

    @Override
    protected double maintain() {
        applicationRepository.deleteExpiredLocalSessions();

        if (hostedVespa) {
            Duration expiryTime = Duration.ofMinutes(90);
            int deleted = applicationRepository.deleteExpiredRemoteSessions(expiryTime);
            log.log(Level.FINE, () -> "Deleted " + deleted + " expired remote sessions older than " + expiryTime);
        }

        return 1.0;
    }

}
