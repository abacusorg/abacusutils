#!/usr/bin/env bash

# fail on error
set -e

# change to tests/
cd $(dirname $0)

# run pipe_asdf
TMPFN=$(mktemp)
pipe_asdf Mini_N64_L32/halos/z0.000/halo_info/halo_info_*.asdf -f N -f x_com | ../pipe_asdf/client > $TMPFN

# compare to known output
diff -s $TMPFN ./ref/pipe.txt
