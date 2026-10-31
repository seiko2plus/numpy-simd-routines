#!/bin/sh
# Vendor a Tabulator release from npm into doc/_static/tabulator.
# Usage: tools/update_tabulator.sh [VERSION]   (default: latest)
# Then rebuild the docs with: spin docs -- -E
set -eu

registry=https://registry.npmjs.org/tabulator-tables
dest=$(cd "$(dirname "$0")/.." && pwd)/doc/_static/tabulator
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -sSfL "$registry/${1:-latest}" -o "$tmp/meta.json"
version=$(grep -o '"version":"[^"]*"' "$tmp/meta.json" | head -1 | cut -d'"' -f4)
integrity=$(grep -o '"integrity":"[^"]*"' "$tmp/meta.json" | head -1 | cut -d'"' -f4)

curl -sSfL "$registry/-/tabulator-tables-$version.tgz" -o "$tmp/pkg.tgz"
actual=sha512-$(openssl dgst -sha512 -binary "$tmp/pkg.tgz" | openssl base64 -A)
if [ "$actual" != "$integrity" ]; then
    echo "tarball does not match the registry's sha512" >&2
    exit 1
fi

tar -xzf "$tmp/pkg.tgz" -C "$tmp" \
    package/dist/js/tabulator.min.js package/dist/css/tabulator.min.css package/LICENSE
mkdir -p "$dest"
cp "$tmp/package/dist/js/tabulator.min.js" "$tmp/package/dist/css/tabulator.min.css" \
    "$tmp/package/LICENSE" "$dest/"
echo "Tabulator $version -> $dest"
