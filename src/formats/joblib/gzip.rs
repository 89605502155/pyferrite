//! Just enough gzip (RFC 1952) framing to unwrap `joblib.dump(compress='gzip')`.

use crate::error::{Error, Result};
use crate::util::decompress_raw;

/// Strip the gzip header/footer and inflate the enclosed deflate stream.
pub fn decode(buf: &[u8]) -> Result<Vec<u8>> {
    if buf.len() < 18 || buf[0] != 0x1f || buf[1] != 0x8b {
        return Err(Error::format("not a gzip stream"));
    }
    if buf[2] != 8 {
        return Err(Error::unsupported("gzip compression method != deflate"));
    }
    let flg = buf[3];
    let mut p = 10usize;
    if flg & 0x04 != 0 {
        let xlen = u16::from_le_bytes([buf[p], buf[p + 1]]) as usize;
        p += 2 + xlen;
    }
    if flg & 0x08 != 0 {
        while p < buf.len() && buf[p] != 0 {
            p += 1;
        }
        p += 1;
    }
    if flg & 0x10 != 0 {
        while p < buf.len() && buf[p] != 0 {
            p += 1;
        }
        p += 1;
    }
    if flg & 0x02 != 0 {
        p += 2;
    }
    let isize_ = u32::from_le_bytes([
        buf[buf.len() - 4],
        buf[buf.len() - 3],
        buf[buf.len() - 2],
        buf[buf.len() - 1],
    ]) as usize;
    decompress_raw(&buf[p..buf.len() - 8], isize_.max(1 << 16))
}

/// `true` when the buffer looks like one of the containers joblib may produce.
pub fn kind(buf: &[u8]) -> Option<&'static str> {
    match buf {
        [0x1f, 0x8b, ..] => Some("gzip"),
        [0x78, _, ..] => Some("zlib"),
        [b'B', b'Z', b'h', ..] => Some("bz2"),
        [0xfd, b'7', b'z', b'X', b'Z', ..] => Some("xz"),
        [0x04, 0x22, 0x4d, 0x18, ..] => Some("lz4"),
        _ => None,
    }
}
