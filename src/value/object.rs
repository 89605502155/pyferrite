//! Inert capture of Python objects the crate does not model natively.

use super::{Dict, Value};

/// A Python object recorded structurally instead of executed.
///
/// The pickle machine never calls Python, so anything outside the numpy/torch
/// allow-list ends up here with its constructor arguments and `__setstate__`
/// payload preserved. That is enough to inspect scikit-learn estimators,
/// custom `nn.Module` wrappers, and to write the object back out unchanged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyObject {
    /// e.g. `"sklearn.linear_model._base"`.
    pub module: String,
    /// e.g. `"LinearRegression"`.
    pub name: String,
    /// Positional arguments captured from `REDUCE` / `NEWOBJ`.
    pub args: Vec<Value>,
    /// Keyword arguments captured from `NEWOBJ_EX`.
    pub kwargs: Dict,
    /// Payload from `BUILD` (`__setstate__` / `__dict__`).
    pub state: Option<Box<Value>>,
    /// Items appended through `APPEND`/`APPENDS` (list-like objects).
    pub list_items: Vec<Value>,
    /// Items set through `SETITEM`/`SETITEMS` (dict-like objects).
    pub dict_items: Dict,
}

impl PyObject {
    /// Create a new value.
    pub fn new(module: impl Into<String>, name: impl Into<String>) -> Self {
        PyObject { module: module.into(), name: name.into(), ..Default::default() }
    }
    /// Fully qualified name, `module.name`.
    pub fn qualname(&self) -> String {
        if self.module.is_empty() {
            self.name.clone()
        } else {
            format!("{}.{}", self.module, self.name)
        }
    }
    /// Attribute lookup inside the captured `__dict__` state.
    pub fn attr(&self, name: &str) -> Option<&Value> {
        match self.state.as_deref() {
            Some(Value::Dict(d)) => d.get(name),
            Some(Value::Tuple(t)) => match t.first() {
                Some(Value::Dict(d)) => d.get(name),
                _ => None,
            },
            _ => None,
        }
    }
    /// Mutable attribute lookup, for editing weights stored on an estimator.
    pub fn attr_mut(&mut self, name: &str) -> Option<&mut Value> {
        match self.state.as_deref_mut() {
            Some(Value::Dict(d)) => d.get_mut(name),
            Some(Value::Tuple(t)) => match t.first_mut() {
                Some(Value::Dict(d)) => d.get_mut(name),
                _ => None,
            },
            _ => None,
        }
    }
}
