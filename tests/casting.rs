//! Configurable numeric types on read and on write.

mod common;
use common::*;
use ndarray::{ArrayD, IxDyn};
use pyferrite::prelude::*;

#[test]
fn float_downcast_on_read() {
    let t = Tmp::new("cast-read");
    let p = t.join("x.npy");
    write(&p, &Value::Array(f64_3x2())).unwrap();
    let v = read_with(&p, &ReadOptions::new().float_as(FloatKind::F32)).unwrap();
    assert_eq!(v.as_array().unwrap().dtype(), DType::F32);
}

#[test]
fn float_downcast_on_write() {
    let t = Tmp::new("cast-write");
    let p = t.join("x.npy");
    write_with(&p, &Value::Array(f64_3x2()), &WriteOptions::new().float_as(FloatKind::F32))
        .unwrap();
    assert_eq!(read(&p).unwrap().as_array().unwrap().dtype(), DType::F32);
}

#[test]
fn integer_width_is_configurable_both_ways() {
    let t = Tmp::new("cast-int");
    let p = t.join("x.npy");
    let small = Array::I64(ArrayD::from_shape_vec(IxDyn(&[3]), vec![-5, 0, 100]).unwrap());
    write_with(&p, &Value::Array(small), &WriteOptions::new().int_as(IntKind::I16)).unwrap();
    assert_eq!(read(&p).unwrap().as_array().unwrap().dtype(), DType::I16);
    let widened = read_with(&p, &ReadOptions::new().int_as(IntKind::I64)).unwrap();
    assert_eq!(widened.as_array().unwrap().dtype(), DType::I64);
}

#[test]
fn eight_bit_floats_are_available_as_a_target() {
    // `f8` here means an 8-*bit* float, not numpy's 8-*byte* naming.
    let a = Array::F32(ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.5, -1.0, 2.0, 448.0]).unwrap());
    let small = a.cast(&DType::F8E4M3, CastPolicy::Saturate).unwrap();
    assert_eq!(small.dtype(), DType::F8E4M3);
    let back = small.cast(&DType::F32, CastPolicy::Strict).unwrap();
    let v = back.as_f32().unwrap();
    assert_eq!(v.as_slice().unwrap(), &[0.5, -1.0, 2.0, 448.0]);
}

#[test]
fn strict_policy_refuses_to_lose_information() {
    let big = Array::I64(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0, 100_000]).unwrap());
    let err = big.cast(&DType::I16, CastPolicy::Strict).unwrap_err();
    assert!(matches!(err, Error::Cast(_)), "got {err}");
}

#[test]
fn saturate_clamps_and_wrap_truncates() {
    let big = Array::I64(ArrayD::from_shape_vec(IxDyn(&[2]), vec![-99_999, 99_999]).unwrap());
    let sat = big.cast(&DType::I16, CastPolicy::Saturate).unwrap();
    assert_eq!(sat.as_i16().unwrap().as_slice().unwrap(), &[i16::MIN, i16::MAX]);
    let wrapped = big.cast(&DType::I16, CastPolicy::Wrap).unwrap();
    assert_eq!(
        wrapped.as_i16().unwrap().as_slice().unwrap(),
        &[(-99_999i64) as i16, 99_999i64 as i16]
    );
}

#[test]
fn half_precision_survives_a_narrowing_round_trip() {
    let a = Array::F64(ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.5, -0.25, 1024.0]).unwrap());
    let half = a.cast(&DType::F16, CastPolicy::Strict).unwrap();
    let back = half.cast(&DType::F64, CastPolicy::Strict).unwrap();
    assert_eq!(back.as_f64().unwrap().as_slice().unwrap(), &[1.0, 0.5, -0.25, 1024.0]);
}

#[test]
fn bfloat16_keeps_the_exponent_range_of_f32() {
    let a = Array::F32(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0e38, 1.0e-38]).unwrap());
    let bf = a.cast(&DType::BF16, CastPolicy::Strict).unwrap();
    let back = bf.cast(&DType::F32, CastPolicy::Strict).unwrap();
    let v = back.as_f32().unwrap();
    assert!(v.as_slice().unwrap()[0].is_finite(), "bf16 should not overflow at 3e38");
    assert!(v.as_slice().unwrap()[1] > 0.0, "bf16 should not flush 1e-38 to zero");
}

#[test]
fn casts_apply_through_a_whole_container() {
    let t = Tmp::new("cast-container");
    let p = t.join("m.npz");
    write(&p, &sample_dict()).unwrap();
    // `b` holds 2^40, which does not fit in an i32, so Strict must refuse.
    let strict = ReadOptions::new().float_as(FloatKind::F32).int_as(IntKind::I32);
    assert!(matches!(read_with(&p, &strict), Err(Error::Cast(_))));
    let v = read_with(&p, &strict.clone().cast_policy(CastPolicy::Saturate)).unwrap();
    for (name, a) in flatten_arrays(&v) {
        match name.as_str() {
            "w" | "x" => assert_eq!(a.dtype(), DType::F32, "{name}"),
            "b" => {
                assert_eq!(a.dtype(), DType::I32, "{name}");
                assert_eq!(a.as_i32().unwrap().as_slice().unwrap()[3], i32::MAX);
            }
            _ => {}
        }
    }
}
