//! Text and integer decoding helpers for protocol-0 pickle opcodes.

use crate::value::Value;

/// Decode a little-endian two's complement `LONG1`/`LONG4` payload.
pub(crate) fn bigint_from_le(b: &[u8]) -> Value {
    if b.is_empty() {
        return Value::Int(0);
    }
    if b.len() <= 8 {
        let mut v: i64 = 0;
        for (i, x) in b.iter().enumerate() {
            v |= (*x as i64) << (8 * i);
        }
        // sign-extend
        let bits = 8 * b.len();
        if bits < 64 && (v >> (bits - 1)) & 1 == 1 {
            v |= -1i64 << bits;
        }
        Value::Int(v)
    } else {
        Value::BigInt(b.to_vec())
    }
}

/// Undo Python's `repr` escaping inside protocol-0 string opcodes.
pub(crate) fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('\'') => out.push('\''),
            Some('"') => out.push('"'),
            Some('x') => {
                let h: String = it.by_ref().take(2).collect();
                if let Ok(v) = u8::from_str_radix(&h, 16) {
                    out.push(v as char);
                }
            }
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                if let Ok(v) = u32::from_str_radix(&h, 16) {
                    out.push(char::from_u32(v).unwrap_or('\u{FFFD}'));
                }
            }
            Some(o) => out.push(o),
            None => break,
        }
    }
    out
}
