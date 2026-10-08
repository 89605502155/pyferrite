//! Insertion-ordered mapping mirroring Python's `dict`.

use super::Value;

/// An ordered `dict`. Keys keep their original Python type, which matters for
/// `state_dict`s whose keys are strings but also for tuple-keyed lookup tables.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dict {
    entries: Vec<(Value, Value)>,
}

impl Dict {
    /// Create a new value.
    pub fn new() -> Self {
        Dict { entries: Vec::new() }
    }
    /// With capacity.
    pub fn with_capacity(n: usize) -> Self {
        Dict { entries: Vec::with_capacity(n) }
    }
    /// Len.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// `true` when this is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Insert or overwrite, preserving the original position on overwrite.
    pub fn insert(&mut self, k: Value, v: Value) {
        if let Some(slot) = self.entries.iter_mut().find(|(key, _)| key == &k) {
            slot.1 = v;
        } else {
            self.entries.push((k, v));
        }
    }
    /// Look up by string key (the common case for `state_dict`s).
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v)
    }
    /// Mutable lookup by string key — the entry point for in-place rescaling.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.entries.iter_mut().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v)
    }
    /// Look up by an arbitrary key value.
    pub fn get_by(&self, key: &Value) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    /// Remove.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let idx = self.entries.iter().position(|(k, _)| k.as_str() == Some(key))?;
        Some(self.entries.remove(idx).1)
    }
    /// Contains key.
    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
    /// Iter.
    pub fn iter(&self) -> std::slice::Iter<'_, (Value, Value)> {
        self.entries.iter()
    }
    /// Iter mut.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, (Value, Value)> {
        self.entries.iter_mut()
    }
    /// Every key that is a string, in insertion order.
    pub fn keys(&self) -> Vec<String> {
        self.entries.iter().filter_map(|(k, _)| k.as_str().map(|s| s.to_string())).collect()
    }
    /// Values mut.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.entries.iter_mut().map(|(_, v)| v)
    }
    /// Take the contents as entries, when the type matches.
    pub fn into_entries(self) -> Vec<(Value, Value)> {
        self.entries
    }
}

impl FromIterator<(Value, Value)> for Dict {
    fn from_iter<T: IntoIterator<Item = (Value, Value)>>(it: T) -> Self {
        Dict { entries: it.into_iter().collect() }
    }
}

impl<'a> IntoIterator for &'a Dict {
    type Item = &'a (Value, Value);
    type IntoIter = std::slice::Iter<'a, (Value, Value)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}
