// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.search.result;

import com.yahoo.api.annotations.Beta;

/**
 * The coverage report for a result set.
 *
 * @author Steinar Knutsen
 * @author baldersheim
 */
public class Coverage extends com.yahoo.container.handler.Coverage {

    public Coverage(long docs, long active) {
        this(docs, active, docs > 1 ? 1 : 0, docs > 1 ? 1: 0);
    }

    public Coverage(long docs, long active, int nodes) {
        super(docs, active, nodes, 1);
    }

    public Coverage(long docs, long active, int nodes, int resultSets) {
        super(docs, active, nodes, resultSets);
    }

    /**
     * Will set number of documents present in ideal state
     *
     * @param soonActive Number of documents active in ideal state
     * @return self for chaining
     */
    @Beta
    public Coverage setSoonActive(long soonActive) { this.soonActive = soonActive; return this; }

    /**
     * Will set the reasons for degraded coverage as reported by vespa backend.
     *
     * @param degradedReason Reason for degradation
     * @return self for chaining
     */
    public Coverage setDegradedReason(int degradedReason) { this.degradedReason = degradedReason; return this; }

    public Coverage setNodesTried(int nodesTried) { super.setNodesTried(nodesTried); return this; }
}
