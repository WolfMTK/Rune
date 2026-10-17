pub const INITIAL_CAPACITY: usize = 256;

pub const DEC_DIGITS: &[u8; 200] = b"0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

pub const HEX: &[u8; 16] = b"0123456789abcdef";

pub const NEEDS_ESCAPE: [bool; 256] = {
    let mut table = [false; 256];
    let mut i = 0;
    while i < 0x20 {
        table[i] = true;
        i += 1;
    }
    table[b'"' as usize] = true;
    table[b'\\' as usize] = true;
    table
};
