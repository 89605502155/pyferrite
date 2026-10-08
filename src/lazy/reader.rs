//! The lazy reading front-end.

use super::Chunk;
use crate::error::Result;

/// A source that yields [`Chunk`]s until it is exhausted.
///
/// Implemented by every format that supports streaming; users normally hold a
/// [`LazyReader`] instead of naming the concrete type.
pub trait ChunkSource: Send {
    /// Produce the next chunk, or `Ok(None)` at end of stream.
    fn next_chunk(&mut self) -> Result<Option<Chunk>>;
    /// Names of the logical arrays this source will emit, when known upfront.
    fn names(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Type-erased, iterator-shaped handle over any [`ChunkSource`].
///
/// ```no_run
/// # use pyferrite::{read_lazy, ReadOptions};
/// let mut r = read_lazy("weights.npy", &ReadOptions::new().chunk_elements(65_536))?;
/// while let Some(mut chunk) = r.next_chunk()? {
///     // `chunk.data` is owned and mutable: rescale in place, then forward it.
///     if let Some(v) = chunk.data.as_f32_mut() { v.iter_mut().for_each(|x| *x *= 0.5); }
/// }
/// # Ok::<(), pyferrite::Error>(())
/// ```
pub struct LazyReader {
    inner: Box<dyn ChunkSource>,
    done: bool,
}

impl LazyReader {
    /// Wrap a concrete source.
    pub fn new(inner: Box<dyn ChunkSource>) -> Self {
        LazyReader { inner, done: false }
    }
    /// Names of the arrays that will be streamed, when the container knows them.
    pub fn names(&self) -> Vec<String> {
        self.inner.names()
    }
    /// Pull the next chunk explicitly.
    pub fn next_chunk(&mut self) -> Result<Option<Chunk>> {
        if self.done {
            return Ok(None);
        }
        let c = self.inner.next_chunk()?;
        if c.is_none() {
            self.done = true;
        }
        Ok(c)
    }
    /// Drain every remaining chunk into a vector (defeats the point, but handy
    /// in tests).
    pub fn collect_chunks(&mut self) -> Result<Vec<Chunk>> {
        let mut out = Vec::new();
        while let Some(c) = self.next_chunk()? {
            out.push(c);
        }
        Ok(out)
    }
}

impl Iterator for LazyReader {
    type Item = Result<Chunk>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(c)) => Some(Ok(c)),
            Ok(None) => None,
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}
