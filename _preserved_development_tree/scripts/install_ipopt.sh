#!/usr/bin/env bash
# IPOPT + HSL MA57 setup (Linux)
#
# Prerequisites:
#   - HSL license from https://licences.stfc.ac.uk/product/coin-hsl
#   - coin-hsl-archive.tar.gz downloaded
#
# If HSL is unavailable, IPOPT will use MUMPS (default).

set -euo pipefail

echo "=== Installing IPOPT dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    gcc g++ gfortran \
    liblapack-dev libblas-dev \
    pkg-config wget

echo "=== Checking for HSL archive ==="
HSL_ARCHIVE="${1:-}"
if [ -n "$HSL_ARCHIVE" ] && [ -f "$HSL_ARCHIVE" ]; then
    echo "HSL archive found: $HSL_ARCHIVE"
    cd /opt
    sudo tar xzf "$HSL_ARCHIVE"
    cd ThirdParty-HSL
    sudo ./configure --prefix=/usr/local
    sudo make -j$(nproc)
    sudo make install
    HSL_FLAG="--with-hsl=/usr/local"
    echo "HSL MA57 installed."
else
    echo "No HSL archive provided. Using MUMPS (default)."
    HSL_FLAG=""
fi

echo "=== Installing IPOPT via coinbrew ==="
cd /tmp
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
./coinbrew fetch Ipopt --no-prompt
./coinbrew build Ipopt --prefix=/usr/local $HSL_FLAG --no-prompt
sudo ldconfig

echo "=== Verifying installation ==="
python -c "import pandapower; print('pandapower OK')"

echo "=== IPOPT installation complete ==="
