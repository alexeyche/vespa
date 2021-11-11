// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.controller.routing;

import com.google.common.collect.ImmutableSortedSet;
import com.yahoo.config.provision.HostName;
import com.yahoo.config.provision.SystemName;
import com.yahoo.config.provision.zone.RoutingMethod;
import com.yahoo.text.Text;
import com.yahoo.vespa.hosted.controller.api.identifiers.DeploymentId;
import com.yahoo.vespa.hosted.controller.api.integration.zone.ZoneRegistry;
import com.yahoo.vespa.hosted.controller.application.Endpoint;
import com.yahoo.vespa.hosted.controller.application.Endpoint.Port;
import com.yahoo.vespa.hosted.controller.application.EndpointId;
import com.yahoo.vespa.hosted.controller.application.SystemApplication;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Represents the DNS routing policy for a {@link com.yahoo.vespa.hosted.controller.application.Deployment}.
 *
 * @author mortent
 * @author mpolden
 */
public class RoutingPolicy {

    private final RoutingPolicyId id;
    private final HostName canonicalName;
    private final Optional<String> dnsZone;
    private final Set<EndpointId> instanceEndpoints;
    private final Set<EndpointId> applicationEndpoints;
    private final Status status;

    /** DO NOT USE. Public for serialization purposes */
    public RoutingPolicy(RoutingPolicyId id, HostName canonicalName, Optional<String> dnsZone,
                         Set<EndpointId> instanceEndpoints, Set<EndpointId> applicationEndpoints, Status status) {
        this.id = Objects.requireNonNull(id, "id must be non-null");
        this.canonicalName = Objects.requireNonNull(canonicalName, "canonicalName must be non-null");
        this.dnsZone = Objects.requireNonNull(dnsZone, "dnsZone must be non-null");
        this.instanceEndpoints = ImmutableSortedSet.copyOf(Objects.requireNonNull(instanceEndpoints, "instanceEndpoints must be non-null"));
        this.applicationEndpoints = ImmutableSortedSet.copyOf(Objects.requireNonNull(applicationEndpoints, "applicationEndpoints must be non-null"));
        this.status = Objects.requireNonNull(status, "status must be non-null");
    }

    /** The ID of this */
    public RoutingPolicyId id() {
        return id;
    }

    /** The canonical name for the load balancer this applies to (rhs of a CNAME or ALIAS record) */
    public HostName canonicalName() {
        return canonicalName;
    }

    /** DNS zone for the load balancer this applies to, if any. Used when creating ALIAS records. */
    public Optional<String> dnsZone() {
        return dnsZone;
    }

    /** The instance-level endpoints this participates in */
    public Set<EndpointId> instanceEndpoints() {
        return instanceEndpoints;
    }

    /** The application-level endpoints  this participates in */
    public Set<EndpointId> applicationEndpoints() {
        return applicationEndpoints;
    }

    /** Returns the status of this */
    public Status status() {
        return status;
    }

    /** Returns whether this policy applies to given deployment */
    public boolean appliesTo(DeploymentId deployment) {
        return id.owner().equals(deployment.applicationId()) &&
               id.zone().equals(deployment.zoneId());
    }

    /** Returns a copy of this with status set to given status */
    public RoutingPolicy with(Status status) {
        return new RoutingPolicy(id, canonicalName, dnsZone, instanceEndpoints, applicationEndpoints, status);
    }

    /** Returns the zone endpoints of this */
    public List<Endpoint> zoneEndpointsIn(SystemName system, RoutingMethod routingMethod, ZoneRegistry zoneRegistry) {
        Optional<Endpoint> infraEndpoint = SystemApplication.matching(id.owner())
                                                            .flatMap(app -> app.endpointIn(id.zone(), zoneRegistry));
        if (infraEndpoint.isPresent()) {
            return List.of(infraEndpoint.get());
        }
        DeploymentId deployment = new DeploymentId(id.owner(), id.zone());
        List<Endpoint> endpoints = new ArrayList<>();
        endpoints.add(endpoint(routingMethod).target(id.cluster(), deployment).in(system));
        // Add legacy endpoints
        if (routingMethod == RoutingMethod.shared) {
            endpoints.add(endpoint(routingMethod).target(id.cluster(), deployment)
                                                 .on(Port.plain(4080))
                                                 .legacy()
                                                 .in(system));
            endpoints.add(endpoint(routingMethod).target(id.cluster(), deployment)
                                                 .on(Port.tls(4443))
                                                 .legacy()
                                                 .in(system));
        }
        return endpoints;
    }

    /** Returns the region endpoint of this */
    public Endpoint regionEndpointIn(SystemName system, RoutingMethod routingMethod) {
        return endpoint(routingMethod).targetRegion(id.cluster(), id.zone()).in(system);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RoutingPolicy that = (RoutingPolicy) o;
        return id.equals(that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return Text.format("%s [instance endpoints: %s, application endpoints: %s%s], %s owned by %s, in %s", canonicalName,
                           instanceEndpoints, applicationEndpoints,
                           dnsZone.map(z -> ", DNS zone: " + z).orElse(""), id.cluster(), id.owner().toShortString(),
                           id.zone().value());
    }

    private Endpoint.EndpointBuilder endpoint(RoutingMethod routingMethod) {
        return Endpoint.of(id.owner())
                       .on(Port.fromRoutingMethod(routingMethod))
                       .routingMethod(routingMethod);
    }

    /** The status of a routing policy */
    public static class Status {

        private final boolean active;
        private final RoutingStatus routingStatus;

        /** DO NOT USE. Public for serialization purposes */
        public Status(boolean active, RoutingStatus routingStatus) {
            this.active = active;
            this.routingStatus = Objects.requireNonNull(routingStatus, "globalRouting must be non-null");
        }

        /** Returns whether this is considered active according to the load balancer status */
        public boolean isActive() {
            return active;
        }

        /** Return status of routing */
        public RoutingStatus routingStatus() {
            return routingStatus;
        }

        /** Returns a copy of this with routing status changed */
        public Status with(RoutingStatus routingStatus) {
            return new Status(active, routingStatus);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Status status = (Status) o;
            return active == status.active &&
                   routingStatus.equals(status.routingStatus);
        }

        @Override
        public int hashCode() {
            return Objects.hash(active, routingStatus);
        }

    }

}
