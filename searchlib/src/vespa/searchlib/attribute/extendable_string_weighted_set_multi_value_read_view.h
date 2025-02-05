// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/searchcommon/attribute/i_multi_value_read_view.h>

namespace search::attribute {

/**
 * Read view for the data stored in an extendable multi-value string
 * weighted set attribute vector (used by streaming visitor) that handles
 * optional removal of weight.
 * @tparam MultiValueType The multi-value type of the data to access.
 */
template <typename MultiValueType>
class ExtendableStringWeightedSetMultiValueReadView : public attribute::IMultiValueReadView<MultiValueType>
{
    const std::vector<char>&            _buffer;
    const vespalib::Array<uint32_t>&    _offsets;
    const std::vector<uint32_t>&        _idx;
    const std::vector<int32_t>&         _weights;
    mutable std::vector<MultiValueType> _copy;
public:
    ExtendableStringWeightedSetMultiValueReadView(const std::vector<char>& buffer, const vespalib::Array<uint32_t>& offsets, const std::vector<uint32_t>& idx, const std::vector<int32_t>& weights);
    ~ExtendableStringWeightedSetMultiValueReadView() override;
    vespalib::ConstArrayRef<MultiValueType> get_values(uint32_t doc_id) const override;
};

}
