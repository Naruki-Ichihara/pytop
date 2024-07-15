#!/bin/bash
rm -rf docs
pdoc --html pytop --output-dir docs
mv docs/pytop/* docs/
rmdir docs/pytop