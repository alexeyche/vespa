// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.controller.restapi.billing;

import com.yahoo.config.provision.SystemName;
import com.yahoo.config.provision.TenantName;
import com.yahoo.vespa.hosted.controller.api.integration.billing.Bill;
import com.yahoo.vespa.hosted.controller.api.integration.billing.CollectionMethod;
import com.yahoo.vespa.hosted.controller.api.integration.billing.MockBillingController;
import com.yahoo.vespa.hosted.controller.api.integration.billing.PlanId;
import com.yahoo.vespa.hosted.controller.api.role.Role;
import com.yahoo.vespa.hosted.controller.restapi.ContainerTester;
import com.yahoo.vespa.hosted.controller.restapi.ControllerContainerCloudTest;
import com.yahoo.vespa.hosted.controller.security.Auth0Credentials;
import com.yahoo.vespa.hosted.controller.security.CloudTenantSpec;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import static com.yahoo.application.container.handler.Request.Method.*;
import static org.junit.Assert.*;

/**
 * @author olaa
 */
public class BillingApiHandlerTest extends ControllerContainerCloudTest {

    private static final String responseFiles = "src/test/java/com/yahoo/vespa/hosted/controller/restapi/billing/responses/";
    private static final TenantName tenant = TenantName.from("tenant1");
    private static final TenantName tenant2 = TenantName.from("tenant2");
    private static final Set<Role> tenantRole = Set.of(Role.administrator(tenant));
    private static final Set<Role> financeAdmin = Set.of(Role.hostedAccountant());
    private MockBillingController billingController;

    private ContainerTester tester;

    @Before
    public void setup() {
        tester = new ContainerTester(container, responseFiles);
        billingController = (MockBillingController) tester.serviceRegistry().billingController();
    }

    @Override
    protected SystemName system() {
        return SystemName.PublicCd;
    }

    @Override
    protected String variablePartXml() {
        return "  <component id='com.yahoo.vespa.hosted.controller.security.CloudAccessControlRequests'/>\n" +
                "  <component id='com.yahoo.vespa.hosted.controller.security.CloudAccessControl'/>\n" +

                "  <handler id='com.yahoo.vespa.hosted.controller.restapi.billing.BillingApiHandler'>\n" +
                "    <binding>http://*/billing/v1/*</binding>\n" +
                "  </handler>\n" +

                "  <http>\n" +
                "    <server id='default' port='8080' />\n" +
                "    <filtering>\n" +
                "      <request-chain id='default'>\n" +
                "        <filter id='com.yahoo.vespa.hosted.controller.restapi.filter.ControllerAuthorizationFilter'/>\n" +
                "        <binding>http://*/*</binding>\n" +
                "      </request-chain>\n" +
                "    </filtering>\n" +
                "  </http>\n";
    }

    @Test
    public void setting_and_deleting_instrument() {
        assertTrue(billingController.getDefaultInstrument(tenant).isEmpty());

        var instrumentRequest = request("/billing/v1/tenant/tenant1/instrument", PATCH)
                .data("{\"active\": \"id-1\"}")
                .roles(tenantRole);

        tester.assertResponse(instrumentRequest,"OK");
        assertEquals("id-1", billingController.getDefaultInstrument(tenant).get().getId());

        var deleteInstrumentRequest = request("/billing/v1/tenant/tenant1/instrument/id-1", DELETE)
                .roles(tenantRole);

        tester.assertResponse(deleteInstrumentRequest,"OK");
        assertTrue(billingController.getDefaultInstrument(tenant).isEmpty());
    }

    @Test
    public void response_list_bills() {
        var bill = createBill();

        billingController.addBill(tenant, bill, true);
        billingController.addBill(tenant, bill, false);
        billingController.setPlan(tenant, PlanId.from("some-plan"), true);

        var request = request("/billing/v1/tenant/tenant1/billing?until=2020-05-28").roles(tenantRole);
        tester.assertResponse(request, new File("tenant-billing-view"));

    }

    @Test
    public void test_bill_creation() {
        var bills = billingController.getBillsForTenant(tenant);
        assertEquals(0, bills.size());

        String requestBody = "{\"tenant\":\"tenant1\", \"startTime\":\"2020-04-20\", \"endTime\":\"2020-05-20\"}";
        var request = request("/billing/v1/invoice", POST)
                .data(requestBody)
                .roles(tenantRole);

        tester.assertResponse(request, accessDenied, 403);
        request.roles(financeAdmin);
        tester.assertResponse(request, new File("invoice-creation-response"));

        bills = billingController.getBillsForTenant(tenant);
        assertEquals(1, bills.size());
        Bill bill = bills.get(0);
        assertEquals("2020-04-20T00:00Z[UTC]", bill.getStartTime().toString());
        assertEquals("2020-05-21T00:00Z[UTC]", bill.getEndTime().toString());

        assertEquals("2020-04-20", bill.getStartDate().toString());
        assertEquals("2020-05-20", bill.getEndDate().toString());
    }

    @Test
    public void adding_and_listing_line_item() {

        var requestBody = "{" +
                "\"description\":\"some description\"," +
                "\"amount\":\"123.45\" " +
                "}";

        var request = request("/billing/v1/invoice/tenant/tenant1/line-item", POST)
                .data(requestBody)
                .roles(financeAdmin);

        tester.assertResponse(request, "{\"message\":\"Added line item for tenant tenant1\"}");

        var lineItems = billingController.getUnusedLineItems(tenant);
        Assert.assertEquals(1, lineItems.size());
        Bill.LineItem lineItem = lineItems.get(0);
        assertEquals("some description", lineItem.description());
        assertEquals(new BigDecimal("123.45"), lineItem.amount());

        request = request("/billing/v1/invoice/tenant/tenant1/line-item")
                .roles(financeAdmin);

        tester.assertResponse(request, new File("line-item-list"));
    }

    @Test
    public void adding_new_status() {
        billingController.addBill(tenant, createBill(), true);

        var requestBody = "{\"status\":\"DONE\"}";
        var request = request("/billing/v1/invoice/id-1/status", POST)
                .data(requestBody)
                .roles(financeAdmin);
        tester.assertResponse(request, "{\"message\":\"Updated status of invoice id-1\"}");

        var bill = billingController.getBillsForTenant(tenant).get(0);
        assertEquals("DONE", bill.status());
    }

    @Test
    public void list_all_unbilled_items() {
        tester.controller().tenants().create(new CloudTenantSpec(tenant, ""), new Auth0Credentials(() -> "foo", Set.of(Role.hostedOperator())));
        tester.controller().tenants().create(new CloudTenantSpec(tenant2, ""), new Auth0Credentials(() -> "foo", Set.of(Role.hostedOperator())));

        var bill = createBill();
        billingController.setPlan(tenant, PlanId.from("some-plan"), true);
        billingController.setPlan(tenant2, PlanId.from("some-plan"), true);
        billingController.addBill(tenant, bill, false);
        billingController.addLineItem(tenant, "support", new BigDecimal("42"), "Smith");
        billingController.addBill(tenant2, bill, false);

        var request = request("/billing/v1/billing?until=2020-05-28").roles(financeAdmin);

        tester.assertResponse(request, new File("billing-all-tenants"));
    }

    @Test
    public void setting_plans() {
        var planRequest = request("/billing/v1/tenant/tenant1/plan", PATCH)
                .data("{\"plan\": \"new-plan\"}")
                .roles(tenantRole);
        tester.assertResponse(planRequest, "Plan: new-plan");
        assertEquals("new-plan", billingController.getPlan(tenant).value());
    }

    @Test
    public void csv_export() {
        var bill = createBill();
        billingController.addBill(tenant, bill, true);
        var csvRequest = request("/billing/v1/invoice/export", GET).roles(financeAdmin);
        tester.assertResponse(csvRequest.get(), new File("billing-all-invoices"), 200, false);
    }

    @Test
    public void patch_collection_method() {
        test_patch_collection_with_field_name("collectionMethod");
        test_patch_collection_with_field_name("collection");
    }

    private void test_patch_collection_with_field_name(String fieldName) {
        var planRequest = request("/billing/v1/tenant/tenant1/collection", PATCH)
                .data("{\"" + fieldName + "\": \"invoice\"}")
                .roles(financeAdmin);
        tester.assertResponse(planRequest, "Collection method updated to INVOICE");
        assertEquals(CollectionMethod.INVOICE, billingController.getCollectionMethod(tenant));

        // Test that not event tenant administrators can do this
        planRequest = request("/billing/v1/tenant/tenant1/collection", PATCH)
                .data("{\"collectionMethod\": \"epay\"}")
                .roles(tenantRole);
        tester.assertResponse(planRequest, accessDenied, 403);
        assertEquals(CollectionMethod.INVOICE, billingController.getCollectionMethod(tenant));
    }

    static Bill createBill() {
        var start = LocalDate.of(2020, 5, 23).atStartOfDay(ZoneOffset.UTC);
        var end = start.toLocalDate().plusDays(6).atStartOfDay(ZoneOffset.UTC);
        var statusHistory = new Bill.StatusHistory(new TreeMap<>(Map.of(start, "OPEN")));
        return new Bill(
                Bill.Id.of("id-1"),
                TenantName.defaultName(),
                statusHistory,
                List.of(createLineItem(start)),
                start,
                end
        );
    }

    static Bill.LineItem createLineItem(ZonedDateTime addedAt) {
        return new Bill.LineItem(
                "some-id",
                "description",
                new BigDecimal("123.00"),
                "some-plan",
                "Smith",
                addedAt
        );
    }

}
