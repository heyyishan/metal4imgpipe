# Creates the deliverable zip file

set -e
cd "$(dirname "$0")"

ZIP_NAME="metal4-imgpipe.zip"

# Clean any previous package
rm -f "$ZIP_NAME"

# Create zip with all source files
zip -r "$ZIP_NAME" \
    CMakeLists.txt \
    README.md \
    setup.py \
    build.sh \
    benchmark.sh \
    src/ \
    tests/ \
    -x "*.DS_Store" "*__pycache__*" "*.pyc"

echo "Created $ZIP_NAME"
ls -lh "$ZIP_NAME"
