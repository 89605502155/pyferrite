//! The Rust-side data model: dicts, frames, objects and exotic numerics.

mod common;
use common::*;
use ndarray::{ArrayD, IxDyn};
use pyferrite::prelude::*;

#[test]
fn dicts_preserve_insertion_order_like_python() {
    let mut d = Dict::new();
    for k in ["zeta", "alpha", "mu"] {
        d.insert(Value::Str(k.into()), Value::Int(1));
    }
    let keys: Vec<_> = d.keys();
    assert_eq!(keys, vec!["zeta", "alpha", "mu"]);
}

#[test]
fn dict_values_are_mutable_in_place() {
    let mut d = Dict::new();
    d.insert(Value::Str("n".into()), Value::Int(1));
    if let Some(Value::Int(n)) = d.get_mut("n") {
        *n = 42;
    }
    assert_eq!(d.get("n").and_then(|v| v.as_i64()), Some(42));
}

#[test]
fn arrays_mut_reaches_every_nested_array() {
    let mut inner = Dict::new();
    inner.insert(Value::Str("a".into()), Value::Array(f32_2x3()));
    let mut outer = Dict::new();
    outer.insert(Value::Str("g".into()), Value::Dict(inner));
    outer.insert(Value::Str("l".into()), Value::List(vec![Value::Array(f32_2x3())]));
    let mut v = Value::Dict(outer);
    let mut n = 0;
    for a in v.arrays_mut() {
        if let Some(f) = a.as_f32_mut() {
            f.iter_mut().for_each(|x| *x += 1.0);
            n += 1;
        }
    }
    assert_eq!(n, 2, "both the nested and the listed array must be reachable");
}

#[test]
fn reshaping_checks_the_element_count() {
    let mut a = f32_2x3();
    a.reshape(&[3, 2]).unwrap();
    assert_eq!(a.shape(), &[3, 2]);
    assert!(a.reshape(&[4, 4]).is_err());
}

#[test]
fn frames_carry_named_columns() {
    let mut f = Frame::default();
    f.push_column(Series::new("x", u8_5())).unwrap();
    f.push_column(Series::new(
        "y",
        Array::F32(ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()),
    ))
    .unwrap();
    // A column of the wrong height must be rejected.
    assert!(f.push_column(Series::new("bad", i64_4())).is_err());
    assert_eq!(f.width(), 2);
    assert_eq!(f.height(), 5);
    assert_eq!(f.names(), vec!["x", "y"]);
    assert!(f.column("y").is_some());
    assert!(f.column("nope").is_none());
}

#[test]
fn f128_arithmetic_beats_f64_precision() {
    let third = F128::from_f64(1.0) / F128::from_f64(3.0);
    let back = third * F128::from_f64(3.0);
    assert_eq!(back.to_f64(), 1.0, "1/3*3 must round-trip exactly in binary128");
}

#[test]
fn f128_survives_the_extremes_of_f64() {
    for v in [1e-300, 1e300, -2.5e-310, f64::MIN_POSITIVE] {
        assert_eq!(F128::from_f64(v).to_f64(), v, "round trip of {v}");
    }
}

#[test]
fn complex_arrays_round_trip_through_every_container() {
    for ext in ["npy", "npz", "pkl"] {
        let t = Tmp::new(&format!("cplx-{ext}"));
        let p = t.join(&format!("c.{ext}"));
        let src = if ext == "npy" {
            Value::Array(c128_2())
        } else {
            let mut d = Dict::new();
            d.insert(Value::Str("z".into()), Value::Array(c128_2()));
            Value::Dict(d)
        };
        write(&p, &src).unwrap();
        let back = read(&p).unwrap();
        let got = flatten_arrays(&back);
        assert_eq!(got.len(), 1, "{ext}");
        assert_same(&c128_2(), got[0].1, ext);
    }
}

#[test]
fn flatten_uses_dotted_paths_like_a_state_dict() {
    let mut inner = Dict::new();
    inner.insert(Value::Str("weight".into()), Value::Array(f32_2x3()));
    let mut outer = Dict::new();
    outer.insert(Value::Str("layer1".into()), Value::Dict(inner));
    let names: Vec<_> = flatten_arrays(&Value::Dict(outer)).into_iter().map(|(k, _)| k).collect();
    assert_eq!(names, vec!["layer1.weight"]);
}

#[test]
fn dtype_sizes_match_numpy() {
    for (dt, size) in [
        (DType::Bool, 1),
        (DType::I8, 1),
        (DType::I16, 2),
        (DType::I32, 4),
        (DType::I64, 8),
        (DType::U64, 8),
        (DType::F16, 2),
        (DType::BF16, 2),
        (DType::F32, 4),
        (DType::F64, 8),
        (DType::F128, 16),
        (DType::C64, 8),
        (DType::C128, 16),
        (DType::C256, 32),
        (DType::F8E4M3, 1),
        (DType::F8E5M2, 1),
    ] {
        assert_eq!(dt.size(), Some(size), "{dt:?}");
    }
}
