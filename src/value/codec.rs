//! Raw buffer <-> [`Array`] codec shared by npy, HDF5, torch storages and joblib.

use super::Array;
use crate::dtype::{DType, Endian};
use crate::error::{Error, Result};
use crate::num::{decode_x87, Complex32, Complex64, BF16, F128, F16, F8E4M3, F8E5M2};
use ndarray::{ArrayD, IxDyn};

macro_rules! decode_prim {
    ($buf:ident, $shape:ident, $swap:ident, $ty:ty, $n:expr, $variant:ident) => {{
        let count: usize = $shape.iter().product();
        let bytes = $buf[..count * $n].chunks_exact($n);
        // Two separate loops so the common, native-order one vectorises.
        let out: Vec<$ty> = if $swap {
            bytes.map(|c| <$ty>::from_be_bytes(c.try_into().unwrap())).collect()
        } else {
            bytes.map(|c| <$ty>::from_le_bytes(c.try_into().unwrap())).collect()
        };
        Array::$variant(
            ArrayD::from_shape_vec(IxDyn($shape), out).map_err(|e| Error::format(e.to_string()))?,
        )
    }};
}

/// Decode a densely packed C-order buffer into an [`Array`].
pub fn from_bytes(dt: &DType, endian: Endian, shape: &[usize], buf: &[u8]) -> Result<Array> {
    let count: usize = shape.iter().product();
    let isize_ = dt.size().ok_or_else(|| Error::unsupported("object dtype has no fixed size"))?;
    if buf.len() < count * isize_ {
        return Err(Error::format(format!(
            "buffer holds {} bytes, need {}",
            buf.len(),
            count * isize_
        )));
    }
    let swap = endian.needs_swap();
    Ok(match dt {
        DType::Bool => Array::Bool(
            ArrayD::from_shape_vec(IxDyn(shape), buf[..count].iter().map(|b| *b != 0).collect())
                .map_err(|e| Error::format(e.to_string()))?,
        ),
        DType::I8 => decode_prim!(buf, shape, swap, i8, 1, I8),
        DType::U8 => decode_prim!(buf, shape, swap, u8, 1, U8),
        DType::I16 => decode_prim!(buf, shape, swap, i16, 2, I16),
        DType::U16 => decode_prim!(buf, shape, swap, u16, 2, U16),
        DType::I32 => decode_prim!(buf, shape, swap, i32, 4, I32),
        DType::U32 => decode_prim!(buf, shape, swap, u32, 4, U32),
        DType::I64 => decode_prim!(buf, shape, swap, i64, 8, I64),
        DType::U64 => decode_prim!(buf, shape, swap, u64, 8, U64),
        DType::F32 => decode_prim!(buf, shape, swap, f32, 4, F32),
        DType::F64 => decode_prim!(buf, shape, swap, f64, 8, F64),
        DType::F8E4M3 => Array::F8E4M3(shaped(shape, buf[..count].iter().map(|b| F8E4M3(*b)))?),
        DType::F8E5M2 => Array::F8E5M2(shaped(shape, buf[..count].iter().map(|b| F8E5M2(*b)))?),
        DType::F16 => Array::F16(shaped(shape, (0..count).map(|i| F16(rd16(buf, i, swap))))?),
        DType::BF16 => Array::BF16(shaped(shape, (0..count).map(|i| BF16(rd16(buf, i, swap))))?),
        DType::F128 => Array::F128(shaped(
            shape,
            (0..count).map(|i| {
                let mut b = [0u8; 16];
                b.copy_from_slice(&buf[i * 16..i * 16 + 16]);
                if swap {
                    b.reverse();
                }
                F128(u128::from_le_bytes(b))
            }),
        )?),
        DType::X87 => Array::F128(shaped(shape, (0..count).map(|i| decode_x87(&buf[i * 16..])))?),
        DType::C64 => Array::C64(shaped(
            shape,
            (0..count)
                .map(|i| Complex32::new(rdf32(buf, 2 * i, swap), rdf32(buf, 2 * i + 1, swap))),
        )?),
        DType::C128 => Array::C128(shaped(
            shape,
            (0..count)
                .map(|i| Complex64::new(rdf64(buf, 2 * i, swap), rdf64(buf, 2 * i + 1, swap))),
        )?),
        DType::C256 => Array::C128(shaped(
            shape,
            (0..count).map(|i| {
                Complex64::new(
                    decode_x87(&buf[i * 32..]).to_f64(),
                    decode_x87(&buf[i * 32 + 16..]).to_f64(),
                )
            }),
        )?),
        DType::Str(n) => Array::Str(shaped(
            shape,
            (0..count).map(|i| {
                let base = i * 4 * n;
                let mut s = String::with_capacity(*n);
                for k in 0..*n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&buf[base + k * 4..base + k * 4 + 4]);
                    if swap {
                        b.reverse();
                    }
                    let cp = u32::from_le_bytes(b);
                    if cp == 0 {
                        break;
                    }
                    s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                }
                s
            }),
        )?),
        DType::Bytes(n) => Array::Bytes(shaped(
            shape,
            (0..count).map(|i| {
                let sl = &buf[i * n..i * n + n];
                let end = sl.iter().rposition(|b| *b != 0).map(|p| p + 1).unwrap_or(0);
                sl[..end].to_vec()
            }),
        )?),
        DType::Object | DType::Struct(_) => {
            return Err(Error::unsupported("object/structured dtype in raw buffer decode"))
        }
    })
}

fn shaped<T, I: Iterator<Item = T>>(shape: &[usize], it: I) -> Result<ArrayD<T>> {
    ArrayD::from_shape_vec(IxDyn(shape), it.collect()).map_err(|e| Error::format(e.to_string()))
}

fn rd16(buf: &[u8], i: usize, swap: bool) -> u16 {
    let mut b = [buf[i * 2], buf[i * 2 + 1]];
    if swap {
        b.reverse();
    }
    u16::from_le_bytes(b)
}
fn rdf32(buf: &[u8], i: usize, swap: bool) -> f32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[i * 4..i * 4 + 4]);
    if swap {
        b.reverse();
    }
    f32::from_le_bytes(b)
}
fn rdf64(buf: &[u8], i: usize, swap: bool) -> f64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[i * 8..i * 8 + 8]);
    if swap {
        b.reverse();
    }
    f64::from_le_bytes(b)
}
