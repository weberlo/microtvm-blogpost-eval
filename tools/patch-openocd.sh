#!/bin/bash -e

cd $(dirname $0)
cd $(git rev-parse --show-toplevel)/3rdparty/openocd

(cat - | git apply --ignore-whitespace -) <<EOF
diff --git a/src/flash/nor/stm32l4x.c b/src/flash/nor/stm32l4x.c
index 2cc378a9..85917db3 100644
--- a/src/flash/nor/stm32l4x.c
+++ b/src/flash/nor/stm32l4x.c
@@ -1038,7 +1038,7 @@ static int stm32l4_probe(struct flash_bank *bank)
        /* in dual bank mode number of pages is doubled, but extra bit is bank selection */
        stm32l4_info->wrpxxr_mask = ((max_pages >> (stm32l4_info->dual_bank_mode ? 1 : 0)) - 1);
        assert((stm32l4_info->wrpxxr_mask & 0xFFFF0000) == 0);
-       LOG_DEBUG("WRPxxR mask 0x%04" PRIx16, stm32l4_info->wrpxxr_mask);
+       LOG_DEBUG("WRPxxR mask 0x%04" PRIx16, (uint16_t) stm32l4_info->wrpxxr_mask);

        if (bank->sectors) {
                free(bank->sectors);
EOF
