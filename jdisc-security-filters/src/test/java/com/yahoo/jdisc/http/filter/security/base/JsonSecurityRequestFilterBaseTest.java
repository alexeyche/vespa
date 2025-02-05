// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.jdisc.http.filter.security.base;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.container.jdisc.RequestHandlerTestDriver;
import com.yahoo.jdisc.Response;
import com.yahoo.jdisc.http.filter.DiscFilterRequest;
import org.junit.Test;

import java.io.IOException;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.mock;

/**
 * @author bjorncs
 */
public class JsonSecurityRequestFilterBaseTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    public void filter_renders_errors_as_json() throws IOException {
        int statusCode = 403;
        String message = "Forbidden";
        DiscFilterRequest request = mock(DiscFilterRequest.class);
        SimpleSecurityRequestFilter filter =
                new SimpleSecurityRequestFilter(new JsonSecurityRequestFilterBase.ErrorResponse(statusCode, message));
        RequestHandlerTestDriver.MockResponseHandler responseHandler = new RequestHandlerTestDriver.MockResponseHandler();
        filter.filter(request, responseHandler);

        Response response = responseHandler.getResponse();
        assertNotNull(response);
        assertEquals(statusCode, response.getStatus());

        JsonNode jsonNode = mapper.readTree(responseHandler.readAll());
        assertEquals(message, jsonNode.get("message").asText());
        assertEquals(statusCode, jsonNode.get("code").asInt());
    }

    private static class SimpleSecurityRequestFilter extends JsonSecurityRequestFilterBase {
        private final ErrorResponse errorResponse;

        SimpleSecurityRequestFilter(ErrorResponse errorResponse) {
            this.errorResponse = errorResponse;
        }

        @Override
        protected Optional<ErrorResponse> filter(DiscFilterRequest request) {
            return Optional.ofNullable(this.errorResponse);
        }
    }

}
