// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.controller.maintenance;

import com.yahoo.config.provision.SystemName;
import com.yahoo.config.provision.TenantName;
import com.yahoo.vespa.flags.InMemoryFlagSource;
import com.yahoo.vespa.flags.PermanentFlags;
import com.yahoo.vespa.hosted.controller.ControllerTester;
import com.yahoo.vespa.hosted.controller.api.integration.billing.PlanId;
import com.yahoo.vespa.hosted.controller.deployment.ApplicationPackageBuilder;
import com.yahoo.vespa.hosted.controller.deployment.DeploymentTester;
import com.yahoo.vespa.hosted.controller.integration.ZoneApiMock;
import com.yahoo.vespa.hosted.controller.tenant.LastLoginInfo;
import com.yahoo.vespa.hosted.controller.tenant.Tenant;
import org.junit.Test;

import java.time.Duration;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author ogronnesby
 */
public class CloudTrialExpirerTest {

    private final ControllerTester tester = new ControllerTester(SystemName.PublicCd);
    private final DeploymentTester deploymentTester = new DeploymentTester(tester);
    private final CloudTrialExpirer expirer = new CloudTrialExpirer(tester.controller(), Duration.ofMinutes(5));

    @Test
    public void expire_inactive_tenant() {
        registerTenant("trial-tenant", "trial", Duration.ofDays(14).plusMillis(1));
        expirer.maintain();
        assertPlan("trial-tenant", "none");
    }

    @Test
    public void tombstone_inactive_none() {
        registerTenant("none-tenant", "none", Duration.ofDays(28).plusMillis(1));
        expirer.maintain();
        assertEquals(Tenant.Type.deleted, tester.controller().tenants().get(TenantName.from("none-tenant"), true).get().type());
    }

    @Test
    public void keep_inactive_nontrial_tenants() {
        registerTenant("not-a-trial-tenant", "pay-as-you-go", Duration.ofDays(30));
        expirer.maintain();
        assertPlan("not-a-trial-tenant", "pay-as-you-go");
    }

    @Test
    public void keep_active_trial_tenants() {
        registerTenant("active-trial-tenant", "trial", Duration.ofHours(14).minusMillis(1));
        expirer.maintain();
        assertPlan("active-trial-tenant", "trial");
    }

    @Test
    public void keep_inactive_exempt_tenants() {
        registerTenant("exempt-trial-tenant", "trial", Duration.ofDays(40));
        ((InMemoryFlagSource) tester.controller().flagSource()).withListFlag(PermanentFlags.EXTENDED_TRIAL_TENANTS.id(), List.of("exempt-trial-tenant"), String.class);
        expirer.maintain();
        assertPlan("exempt-trial-tenant", "trial");
    }

    @Test
    public void keep_inactive_trial_tenants_with_deployments() {
        registerTenant("with-deployments", "trial", Duration.ofDays(30));
        registerDeployment("with-deployments", "my-app", "default");
        expirer.maintain();
        assertPlan("with-deployments", "trial");
    }

    @Test
    public void delete_tenants_with_applications_with_no_deployments() {
        registerTenant("with-apps", "trial", Duration.ofDays(30));
        tester.createApplication("with-apps", "app1", "instance1");
        expirer.maintain();
        assertPlan("with-apps", "none");
        expirer.maintain();
        assertTrue(tester.controller().tenants().get("with-apps").isEmpty());
    }

    private void registerTenant(String tenantName, String plan, Duration timeSinceLastLogin) {
        var name = TenantName.from(tenantName);
        tester.createTenant(tenantName, Tenant.Type.cloud);
        tester.serviceRegistry().billingController().setPlan(name, PlanId.from(plan), false);
        tester.controller().tenants().updateLastLogin(name, List.of(LastLoginInfo.UserLevel.user), tester.controller().clock().instant().minus(timeSinceLastLogin));
    }

    private void registerDeployment(String tenantName, String appName, String instanceName) {
        var app = tester.createApplication(tenantName, appName, instanceName);
        var ctx = deploymentTester.newDeploymentContext(tenantName, appName, instanceName);
        var pkg = new ApplicationPackageBuilder()
                .instances("default")
                .region("aws-us-east-1c")
                .trustDefaultCertificate()
                .build();
        ctx.submit(pkg).deploy();
    }

    private void assertPlan(String tenant, String planId) {
        assertEquals(planId, tester.serviceRegistry().billingController().getPlan(TenantName.from(tenant)).value());
    }

}
