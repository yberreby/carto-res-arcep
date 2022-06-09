#!/bin/bash
mkdir -p raw
echo "Exporting PDF to PNG files. This might take a while..."
pdftoppm -png -rx 900 -ry 900 DEGROUPAGE_v2.pdf raw/map
echo "Done."
