// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/searchlib/common/sortresults.h>
#include <vespa/searchlib/attribute/attribute.h>
#include <vespa/searchlib/attribute/attributeguard.h>
#include <vespa/searchlib/attribute/attributefactory.h>
#include <vespa/searchlib/attribute/attributecontext.h>
#include <vespa/searchlib/attribute/attributemanager.h>
#include <vespa/searchlib/uca/ucaconverter.h>
#include <vespa/vespalib/util/testclock.h>
#include <vespa/vespalib/testkit/testapp.h>
#include <type_traits>
#include <vespa/log/log.h>
LOG_SETUP("multilevelsort_test");

using namespace search;

typedef FastS_SortSpec::VectorRef VectorRef;
typedef IntegerAttributeTemplate<int8_t>   Int8;
typedef IntegerAttributeTemplate<int16_t>  Int16;
typedef IntegerAttributeTemplate<int32_t>  Int32;
typedef IntegerAttributeTemplate<int64_t>  Int64;
typedef FloatingPointAttributeTemplate<float>  Float;
typedef FloatingPointAttributeTemplate<double> Double;
typedef std::map<std::string, AttributeVector::SP > VectorMap;
typedef AttributeVector::SP             AttributePtr;
using search::attribute::Config;
using search::attribute::BasicType;
using search::attribute::CollectionType;

class MultilevelSortTest {
public:
    enum AttrType {
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT,
        DOUBLE,
        STRING,
        RANK,
        DOCID,
        NONE
    };
    struct Spec {
        Spec() : _name("unknown"), _type(NONE), _asc(true) {}
        Spec(const std::string &name, AttrType type) : _name(name), _type(type), _asc(true) {}
        Spec(const std::string &name, AttrType type, bool asc) : _name(name), _type(type), _asc(asc) {}
        std::string _name;
        AttrType _type;
        bool _asc;
    };
private:
    template<typename T>
    T getRandomValue() {
        T min = std::numeric_limits<T>::min();
        T max = std::numeric_limits<T>::max();
        return min + static_cast<T>((max - min) * (((float)rand() / (float)RAND_MAX)));
    }
    template<typename T>
    void fill(IntegerAttribute *attr, uint32_t size, uint32_t unique = 0);
    template<typename T>
    void fill(FloatingPointAttribute *attr, uint32_t size, uint32_t unique = 0);
    void fill(StringAttribute *attr, uint32_t size, const std::vector<std::string> &values);
    template <typename V>
    int compareTemplate(AttributeVector *vector, uint32_t a, uint32_t b);
    int compare(AttributeVector *vector, AttrType type, uint32_t a, uint32_t b);
    void sortAndCheck(const std::vector<Spec> &spec, uint32_t num,
                      uint32_t unique, const std::vector<std::string> &strValues);
public:
    MultilevelSortTest() { srand(time(nullptr)); }
    void testSort();
};

template<typename T>
void MultilevelSortTest::fill(IntegerAttribute *attr, uint32_t size, uint32_t unique)
{
    ASSERT_TRUE(attr->addDocs(size));
    std::vector<T> values;
    for (uint32_t j = 0; j < unique; ++j) {
        if (j % 2 == 0) {
            values.push_back(std::numeric_limits<T>::min() + static_cast<T>(j));
        } else {
            values.push_back(std::numeric_limits<T>::max() - static_cast<T>(j));
        }
    }
    for (uint32_t i = 0; i < size; ++i) {
        if (unique == 0) {
            attr->update(i, getRandomValue<T>());
        } else {
            uint32_t idx = rand() % values.size();
            attr->update(i, values[idx]);
        }
    }
}

template<typename T>
void MultilevelSortTest::fill(FloatingPointAttribute *attr, uint32_t size, uint32_t unique)
{
    ASSERT_TRUE(attr->addDocs(size));
    std::vector<T> values;
    for (uint32_t j = 0; j < unique; ++j) {
        if (j % 2 == 0) {
            values.push_back(std::numeric_limits<T>::min() + static_cast<T>(j));
        } else {
            values.push_back(std::numeric_limits<T>::max() - static_cast<T>(j));
        }
    }
    for (uint32_t i = 0; i < size; ++i) {
        if (unique == 0) {
            attr->update(i, getRandomValue<T>());
        } else {
            uint32_t idx = rand() % values.size();
            //LOG(info, "fill vector<%s>::doc<%d> = %f (idx=%d)", attr->getName().c_str(), i, values[idx], idx);
            attr->update(i, values[idx]);
        }
    }
}

void
MultilevelSortTest::fill(StringAttribute *attr, uint32_t size, const std::vector<std::string> &values)
{
    ASSERT_TRUE(attr->addDocs(size));
    for (uint32_t i = 0; i < size; ++i) {
        if (values.empty()) {
            uint32_t len = 1 + static_cast<uint32_t>(127 * (((float)rand() / (float)RAND_MAX)));
            std::string value;
            for (uint32_t j = 0; j < len; ++j) {
                char c = 'a' + static_cast<char>(('Z' - 'a') * (((float)rand() / (float)RAND_MAX)));
                value.append(1, c);
            }
            attr->update(i, value.c_str());
        } else {
            uint32_t idx = rand() % values.size();
            //LOG(info, "fill vector<%s>::doc<%d> = %s (idx=%d)", attr->getName().c_str(),
            //    i, values[idx].c_str(), idx);
            attr->update(i, values[idx].c_str());
        }
    }
}

template <typename V>
V get_helper(AttributeVector *vector, uint32_t doc_id) {
    if constexpr (std::is_floating_point_v<V>) {
        return vector->getFloat(doc_id);
    } else {
        return vector->getInt(doc_id);
    }
}

template <typename V>
int
MultilevelSortTest::compareTemplate(AttributeVector *vector, uint32_t a, uint32_t b)
{
    V va;
    V vb;
    va = get_helper<V>(vector, a);
    vb = get_helper<V>(vector, b);
    if (va == vb) {
        return 0;
    } else if (va < vb) {
        return -1;
    }
    return 1;
}

int
MultilevelSortTest::compare(AttributeVector *vector, AttrType type, uint32_t a, uint32_t b)
{
    if (type == INT8) {
        return compareTemplate<int8_t>(vector, a, b);
    } else if (type == INT16) {
        return compareTemplate<int16_t>(vector, a, b);
    } else if (type == INT32) {
        return compareTemplate<int32_t>(vector, a, b);
    } else if (type == INT64) {
        return compareTemplate<int64_t>(vector, a, b);
    } else if (type == FLOAT) {
        return compareTemplate<float>(vector, a, b);
    } else if (type == DOUBLE) {
        return compareTemplate<double>(vector, a, b);
    } else if (type == STRING) {
        StringAttribute *vString = static_cast<StringAttribute*>(vector);
        const char *va = vString->get(a);
        const char *vb = vString->get(b);
        std::string sa(va);
        std::string sb(vb);
        if (sa == sb) {
            return 0;
        } else if (sa < sb) {
            return -1;
        }
        return 1;
    } else {
        ASSERT_TRUE(false);
        return 0;
    }
}

void
MultilevelSortTest::sortAndCheck(const std::vector<Spec> &spec, uint32_t num,
                                 uint32_t unique, const std::vector<std::string> &strValues)
{
    VectorMap vec;
    // generate attribute vectors
    for (uint32_t i = 0; i < spec.size(); ++i) {
        std::string name = spec[i]._name;
        AttrType type = spec[i]._type;
        if (type == INT8) {
            Config cfg(BasicType::INT8, CollectionType::SINGLE);
            vec[name] = AttributeFactory::createAttribute(name, cfg);
            fill<int8_t>(static_cast<IntegerAttribute *>(vec[name].get()), num, unique);
        } else if (type == INT16) {
            Config cfg(BasicType::INT16, CollectionType::SINGLE);
            vec[name] = AttributeFactory::createAttribute(name, cfg);
            fill<int16_t>(static_cast<IntegerAttribute *>(vec[name].get()), num, unique);
        } else if (type == INT32) {
            Config cfg(BasicType::INT32, CollectionType::SINGLE);
            vec[name] = AttributeFactory::createAttribute(name, cfg);
            fill<int32_t>(static_cast<IntegerAttribute *>(vec[name].get()), num, unique);
        } else if (type == INT64) {
            Config cfg(BasicType::INT64, CollectionType::SINGLE);
            vec[name] = AttributeFactory::createAttribute(name, cfg);
            fill<int64_t>(static_cast<IntegerAttribute *>(vec[name].get()), num, unique);
        } else if (type == FLOAT) {
            Config cfg(BasicType::FLOAT, CollectionType::SINGLE);
            vec[name] = AttributeFactory::createAttribute(name, cfg);
            fill<float>(static_cast<FloatingPointAttribute *>(vec[name].get()), num, unique);
        } else if (type == DOUBLE) {
            Config cfg(BasicType::DOUBLE, CollectionType::SINGLE);
            vec[name] = AttributeFactory::createAttribute(name, cfg);
            fill<double>(static_cast<FloatingPointAttribute *>(vec[name].get()), num, unique);
        } else if (type == STRING) {
            Config cfg(BasicType::STRING, CollectionType::SINGLE);
            vec[name] = AttributeFactory::createAttribute(name, cfg);
            fill(static_cast<StringAttribute *>(vec[name].get()), num, strValues);
        }
        if (vec[name])
            vec[name]->commit();
    }

    RankedHit *hits = new RankedHit[num];
    for (uint32_t i = 0; i < num; ++i) {
        hits[i]._docId = i;
        hits[i]._rankValue = getRandomValue<uint32_t>();
    }

    vespalib::TestClock clock;
    vespalib::Doom doom(clock.clock(), vespalib::steady_time::max());
    search::uca::UcaConverterFactory ucaFactory;
    FastS_SortSpec sorter(7, doom, ucaFactory);
    // init sorter with sort data
    for(uint32_t i = 0; i < spec.size(); ++i) {
        AttributeGuard ag;
        if (spec[i]._type == RANK) {
            sorter._vectors.push_back
                (VectorRef(spec[i]._asc ? FastS_SortSpec::ASC_RANK :
                           FastS_SortSpec::DESC_RANK, nullptr, nullptr));
        } else if (spec[i]._type == DOCID) {
            sorter._vectors.push_back
                (VectorRef(spec[i]._asc ? FastS_SortSpec::ASC_DOCID :
                           FastS_SortSpec::DESC_DOCID, nullptr, nullptr));
        } else {
            const search::attribute::IAttributeVector * v = vec[spec[i]._name].get();
            sorter._vectors.push_back
                (VectorRef(spec[i]._asc ? FastS_SortSpec::ASC_VECTOR :
                           FastS_SortSpec::DESC_VECTOR, v, nullptr));
        }
    }

    vespalib::Timer timer;
    sorter.sortResults(hits, num, num);
    LOG(info, "sort time = %" PRId64 " ms", vespalib::count_ms(timer.elapsed()));

    uint32_t *offsets = new uint32_t[num + 1];
    char *buf = new char[sorter.getSortDataSize(0, num)];
    sorter.copySortData(0, num, offsets, buf);

    // check results
    for (uint32_t i = 0; i < num - 1; ++i) {
        for (uint32_t j = 0; j < spec.size(); ++j) {
            int cmp = 0;
            if (spec[j]._type == RANK) {
                if (hits[i].getRank() < hits[i+1].getRank()) {
                    cmp = -1;
                } else if (hits[i].getRank() > hits[i+1].getRank()) {
                    cmp = 1;
                }
            } else if (spec[j]._type == DOCID) {
                if (hits[i].getDocId() < hits[i+1].getDocId()) {
                    cmp = -1;
                } else if (hits[i].getDocId() > hits[i+1].getDocId()) {
                    cmp = 1;
                }
            } else {
                AttributeVector *av = vec[spec[j]._name].get();
                cmp = compare(av, spec[j]._type,
                              hits[i].getDocId(), hits[i+1].getDocId());
            }
            if (spec[j]._asc) {
                EXPECT_TRUE(cmp <= 0);
                if (cmp < 0) {
                    break;
                }
            } else {
                EXPECT_TRUE(cmp >= 0);
                if (cmp > 0) {
                    break;
                }
            }
        }
        // check binary sort data
        uint32_t minLen = std::min(sorter._sortDataArray[i]._len,
                          sorter._sortDataArray[i+1]._len);
        int cmp = memcmp(&sorter._binarySortData[0] + sorter._sortDataArray[i]._idx,
                         &sorter._binarySortData[0] + sorter._sortDataArray[i+1]._idx,
                         minLen);
        EXPECT_TRUE(cmp <= 0);
        EXPECT_TRUE(sorter._sortDataArray[i]._len == (offsets[i+1] - offsets[i]));
        cmp = memcmp(&sorter._binarySortData[0] + sorter._sortDataArray[i]._idx,
                     buf + offsets[i], sorter._sortDataArray[i]._len);
        EXPECT_TRUE(cmp == 0);
    }
    EXPECT_TRUE(sorter._sortDataArray[num-1]._len == (offsets[num] - offsets[num-1]));
    int cmp = memcmp(&sorter._binarySortData[0] + sorter._sortDataArray[num-1]._idx,
                 buf + offsets[num-1], sorter._sortDataArray[num-1]._len);
    EXPECT_TRUE(cmp == 0);

    delete [] hits;
    delete [] offsets;
    delete [] buf;
}

void MultilevelSortTest::testSort()
{
    {
        std::vector<Spec> spec;
        spec.push_back(Spec("int8", INT8));
        spec.push_back(Spec("int16", INT16));
        spec.push_back(Spec("int32", INT32));
        spec.push_back(Spec("int64", INT64));
        spec.push_back(Spec("float", FLOAT));
        spec.push_back(Spec("double", DOUBLE));
        spec.push_back(Spec("string", STRING));
        spec.push_back(Spec("rank", RANK));
        spec.push_back(Spec("docid", DOCID));

        std::vector<std::string> strValues;
        strValues.push_back("applications");
        strValues.push_back("places");
        strValues.push_back("system");
        strValues.push_back("vespa search core");

        srand(12345);
        sortAndCheck(spec, 5000, 4, strValues);
        srand(time(nullptr));
        sortAndCheck(spec, 5000, 4, strValues);

        strValues.push_back("multilevelsort");
        strValues.push_back("trondheim");
        strValues.push_back("ubuntu");
        strValues.push_back("fastserver4");

        srand(56789);
        sortAndCheck(spec, 5000, 8, strValues);
        srand(time(nullptr));
        sortAndCheck(spec, 5000, 8, strValues);
    }
    {
        std::vector<std::string> none;
        uint32_t num = 50;
        sortAndCheck(std::vector<Spec>(1, Spec("int8", INT8, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("int16", INT16, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("int32", INT32, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("int64", INT64, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("float", FLOAT, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("double", DOUBLE, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("string", STRING, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("rank", RANK, true)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("docid", DOCID, true)), num, 0, none);

        sortAndCheck(std::vector<Spec>(1, Spec("int8", INT8, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("int16", INT16, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("int32", INT32, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("int64", INT64, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("float", FLOAT, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("double", DOUBLE, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("string", STRING, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("rank", RANK, false)), num, 0, none);
        sortAndCheck(std::vector<Spec>(1, Spec("docid", DOCID, false)), num, 0, none);
    }

}

TEST("require that all sort methods behave the same")
{
    MultilevelSortTest test;
    test.testSort();
}

TEST("test that [docid] translates to [lid][paritionid]") {
    vespalib::TestClock clock;
    vespalib::Doom doom(clock.clock(), vespalib::steady_time::max());
    search::uca::UcaConverterFactory ucaFactory;
    FastS_SortSpec asc(7, doom, ucaFactory);
    RankedHit hits[2] = {RankedHit(91, 0.0), RankedHit(3, 2.0)};
    search::AttributeManager mgr;
    search::AttributeContext ac(mgr);
    EXPECT_TRUE(asc.Init("+[docid]", ac));
    asc.initWithoutSorting(hits, 2);
    constexpr uint8_t FIRST_ASC[6] = {0,0,0,91,0,7};
    constexpr uint8_t SECOND_ASC[6] = {0,0,0,3,0,7};
    constexpr uint8_t FIRST_DESC[6] = {255,255,255,255-91,255,255-7};
    constexpr uint8_t SECOND_DESC[6] = {255,255,255,255-3,255,255-7};
    auto sr1 = asc.getSortRef(0);
    EXPECT_EQUAL(6u, sr1.second);
    EXPECT_EQUAL(0, memcmp(FIRST_ASC, sr1.first, 6));
    auto sr2 = asc.getSortRef(1);
    EXPECT_EQUAL(6u, sr2.second);
    EXPECT_EQUAL(0, memcmp(SECOND_ASC, sr2.first, 6));

    FastS_SortSpec desc(7, doom, ucaFactory);
    desc.Init("-[docid]", ac);
    desc.initWithoutSorting(hits, 2);
    sr1 = desc.getSortRef(0);
    EXPECT_EQUAL(6u, sr1.second);
    EXPECT_EQUAL(0, memcmp(FIRST_DESC, sr1.first, 6));
    sr2 = desc.getSortRef(1);
    EXPECT_EQUAL(6u, sr2.second);
    EXPECT_EQUAL(0, memcmp(SECOND_DESC, sr2.first, 6));
}

TEST_MAIN() { TEST_RUN_ALL(); }
