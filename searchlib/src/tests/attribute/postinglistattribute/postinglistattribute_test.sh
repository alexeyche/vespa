#!/bin/bash
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -e
$VALGRIND ./searchlib_postinglistattribute_test_app
rm -rf *.dat
rm -rf *.udat
rm -rf *.idx
rm -rf *.weight
