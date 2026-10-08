//! Cross-format helpers: dtype resolution, cast application, axis permutation.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::options::{ReadOptions, WriteOptions};
use crate::value::{Array, Value};
use ndarray::IxDyn;

/// Apply the target dtype implied by `int_as` / `float_as` to a source dtype.
pub fn resolve_target_dtype_read(src: &DType, opts: &ReadOptions) -> DType {
    if let (true, Some(k)) = (src.is_integer(), opts.int_as) {
        return k.dtype();
    }
    if let (true, Some(k)) = (src.is_float(), opts.float_as) {
        return k.dtype();
    }
    if *src == DType::X87 {
        return DType::F128;
    }
    src.clone()
}

/// Same, for the write direction; `opts.dtype` overrides everything.
pub fn resolve_target_dtype(src: &DType, opts: &WriteOptions) -> DType {
    if let Some(dt) = &opts.dtype {
        return dt.clone();
    }
    if let (true, Some(k)) = (src.is_integer(), opts.int_as) {
        return k.dtype();
    }
    if let (true, Some(k)) = (src.is_float(), opts.float_as) {
        return k.dtype();
    }
    src.clone()
}

/// Cast a freshly decoded array according to the read options.
pub fn apply_cast(a: Array, opts: &ReadOptions) -> Result<Array> {
    let target = resolve_target_dtype_read(&a.dtype(), opts);
    if target == a.dtype() {
        Ok(a)
    } else {
        a.cast(&target, opts.cast_policy)
    }
}

/// Permute array axes (used to normalise Fortran-ordered payloads).
pub fn permute(a: Array, axes: &[usize]) -> Array {
    use crate::value::Array as A;
    macro_rules! p {
        ($x:expr, $v:ident) => {
            A::$v($x.permuted_axes(IxDyn(axes)).as_standard_layout().to_owned())
        };
    }
    match a {
        A::Bool(x) => p!(x, Bool),
        A::I8(x) => p!(x, I8),
        A::I16(x) => p!(x, I16),
        A::I32(x) => p!(x, I32),
        A::I64(x) => p!(x, I64),
        A::U8(x) => p!(x, U8),
        A::U16(x) => p!(x, U16),
        A::U32(x) => p!(x, U32),
        A::U64(x) => p!(x, U64),
        A::F8E4M3(x) => p!(x, F8E4M3),
        A::F8E5M2(x) => p!(x, F8E5M2),
        A::F16(x) => p!(x, F16),
        A::BF16(x) => p!(x, BF16),
        A::F32(x) => p!(x, F32),
        A::F64(x) => p!(x, F64),
        A::F128(x) => p!(x, F128),
        A::C64(x) => p!(x, C64),
        A::C128(x) => p!(x, C128),
        A::Str(x) => p!(x, Str),
        A::Bytes(x) => p!(x, Bytes),
    }
}

/// Coerce a scalar / list value into an [`Array`] so it can be written to a
/// format that only stores arrays.
///
/// This is the validation the user asked for: writing a `str` where numpy
/// expects an array fails loudly instead of producing a broken file.
pub fn value_to_array(v: &Value) -> Result<Array> {
    use ndarray::ArrayD;
    let mk = |data: Vec<f64>| -> Result<Array> {
        let n = data.len();
        Ok(Array::F64(
            ArrayD::from_shape_vec(IxDyn(&[n]), data).map_err(|e| Error::format(e.to_string()))?,
        ))
    };
    Ok(match v {
        Value::Array(a) => a.clone(),
        Value::Float(f) => mk(vec![*f])?,
        Value::Int(i) => Array::I64(
            ArrayD::from_shape_vec(IxDyn(&[] as &[usize]), vec![*i])
                .map_err(|e| Error::format(e.to_string()))?,
        ),
        Value::Bool(b) => Array::Bool(
            ArrayD::from_shape_vec(IxDyn(&[] as &[usize]), vec![*b])
                .map_err(|e| Error::format(e.to_string()))?,
        ),
        Value::List(items) | Value::Tuple(items) => {
            if items.iter().all(|x| matches!(x, Value::Int(_) | Value::Bool(_))) {
                let data: Vec<i64> = items.iter().filter_map(|x| x.as_i64()).collect();
                let n = data.len();
                Array::I64(
                    ArrayD::from_shape_vec(IxDyn(&[n]), data)
                        .map_err(|e| Error::format(e.to_string()))?,
                )
            } else if items.iter().all(|x| x.as_f64().is_some()) {
                mk(items.iter().filter_map(|x| x.as_f64()).collect())?
            } else if items.iter().all(|x| matches!(x, Value::Array(_))) {
                return stack(items);
            } else {
                return Err(Error::invalid(
                    "cannot write a heterogeneous list as a numeric array; \
                     convert it to a Value::Array or a Value::Frame first",
                ));
            }
        }
        other => {
            return Err(Error::invalid(format!(
                "cannot write a `{}` as a numeric array — numpy expects numbers, \
                 not `{}`",
                other.type_name(),
                other.type_name()
            )))
        }
    })
}

/// Stack a list of equally shaped arrays along a new leading axis.
fn stack(items: &[Value]) -> Result<Array> {
    let arrays: Vec<&Array> = items.iter().filter_map(|v| v.as_array()).collect();
    let first = arrays.first().ok_or_else(|| Error::invalid("empty array list"))?;
    let inner = first.shape().to_vec();
    let dt = first.dtype();
    for a in &arrays {
        if a.shape() != inner.as_slice() || a.dtype() != dt {
            return Err(Error::invalid("stacked arrays must share dtype and shape"));
        }
    }
    let mut flat: Vec<u8> = Vec::new();
    for a in &arrays {
        flat.extend(crate::value::to_bytes(a, crate::dtype::Endian::Little)?);
    }
    let mut shape = vec![arrays.len()];
    shape.extend(inner);
    crate::value::from_bytes(&dt, crate::dtype::Endian::Little, &shape, &flat)
}
