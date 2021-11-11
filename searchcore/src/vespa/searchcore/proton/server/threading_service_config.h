// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#pragma once

#include <vespa/searchcore/config/config-proton.h>
#include <vespa/searchcore/proton/common/hw_info.h>
#include <vespa/vespalib/util/executor.h>
#include <vespa/vespalib/util/time.h>
#include <cstdint>

namespace vespa::config::search::core::internal { class InternalProtonType; }
namespace proton {

/**
 * Config for the threading service used by a documentdb.
 */
class ThreadingServiceConfig {
public:
    using ProtonConfig = const vespa::config::search::core::internal::InternalProtonType;
    using OptimizeFor = vespalib::Executor::OptimizeFor;
    using SharedFieldWriterExecutor = ProtonConfig::Feeding::SharedFieldWriterExecutor;

private:
    uint32_t           _indexingThreads;
    uint32_t           _defaultTaskLimit;
    OptimizeFor        _optimize;
    uint32_t           _kindOfWatermark;
    vespalib::duration _reactionTime;         // Maximum reaction time to new tasks
    SharedFieldWriterExecutor _shared_field_writer;

private:
    ThreadingServiceConfig(uint32_t indexingThreads_, uint32_t defaultTaskLimit_, OptimizeFor optimize_,
                           uint32_t kindOfWatermark_, vespalib::duration reactionTime_, SharedFieldWriterExecutor shared_field_writer_);

public:
    static ThreadingServiceConfig make(const ProtonConfig &cfg, double concurrency, const HwInfo::Cpu &cpuInfo);
    static ThreadingServiceConfig make(uint32_t indexingThreads, SharedFieldWriterExecutor shared_field_writer_ = SharedFieldWriterExecutor::NONE);
    void update(const ThreadingServiceConfig& cfg);
    uint32_t indexingThreads() const { return _indexingThreads; }
    uint32_t defaultTaskLimit() const { return _defaultTaskLimit; }
    OptimizeFor optimize() const { return _optimize; }
    uint32_t kindOfwatermark() const { return _kindOfWatermark; }
    vespalib::duration reactionTime() const { return _reactionTime; }
    SharedFieldWriterExecutor shared_field_writer() const { return _shared_field_writer; }
    bool operator==(const ThreadingServiceConfig &rhs) const;
};

}
