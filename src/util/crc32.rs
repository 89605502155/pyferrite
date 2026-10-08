//! CRC-32/ISO-HDLC, needed by the zip container (npz, pt).

/// Lazily built lookup table (const-evaluated, so no runtime initialisation).
const TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// CRC-32 of `data` with the standard `0xFFFFFFFF` pre/post conditioning.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for b in data {
        c = TABLE[((c ^ *b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Incremental CRC-32 state for streaming writers.
#[derive(Clone, Copy, Debug)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Crc32 { state: 0xFFFF_FFFF }
    }
}

impl Crc32 {
    /// Start a fresh checksum.
    pub fn new() -> Self {
        Self::default()
    }
    /// Fold another slice of bytes into the running checksum.
    pub fn update(&mut self, data: &[u8]) {
        for b in data {
            self.state = TABLE[((self.state ^ *b as u32) & 0xFF) as usize] ^ (self.state >> 8);
        }
    }
    /// Finalise and return the CRC-32 value.
    pub fn finish(self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}
