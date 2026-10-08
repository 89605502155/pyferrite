//! Materialising dataset bytes from contiguous, compact or chunked storage.

use super::chunks::{btree_chunks, scatter};
use super::msg::{self, Layout};
use super::objhdr::Message;
use super::superblock::Superblock;
use crate::dtype::{DType, Endian};
use crate::error::{Error, Result};
use crate::util::decompress_zlib;

/// Everything needed to decode one dataset.
#[derive(Clone, Debug)]
pub struct DatasetInfo {
    pub dtype: DType,
    pub endian: Endian,
    pub shape: Vec<usize>,
    pub layout: Layout,
    pub filters: Vec<u16>,
}

/// Extract dataset metadata from a parsed object header.
pub fn info(msgs: &[Message], sb: &Superblock) -> Result<Option<DatasetInfo>> {
    let mut shape = None;
    let mut dt = None;
    let mut layout = None;
    let mut filters = Vec::new();
    for m in msgs {
        match m.kind {
            msg::MSG_DATASPACE => shape = Some(msg::dataspace(&m.data, sb)?),
            msg::MSG_DATATYPE => dt = Some(msg::datatype(&m.data)?),
            msg::MSG_LAYOUT => layout = Some(msg::layout(&m.data, sb)?),
            msg::MSG_FILTER => filters = msg::filters(&m.data)?,
            _ => {}
        }
    }
    match (shape, dt, layout) {
        (Some(shape), Some((dtype, endian)), Some(layout)) => {
            Ok(Some(DatasetInfo { dtype, endian, shape, layout, filters }))
        }
        _ => Ok(None),
    }
}

fn unfilter(mut buf: Vec<u8>, filters: &[u16], expect: usize, esz: usize) -> Result<Vec<u8>> {
    for f in filters.iter().rev() {
        buf = match f {
            1 => decompress_zlib(&buf, expect)?,
            2 => unshuffle(&buf, esz),
            3 => {
                buf.truncate(buf.len().saturating_sub(4)); // fletcher32 checksum
                buf
            }
            other => {
                return Err(Error::unsupported(format!(
                    "HDF5 filter id {other} (only deflate, shuffle and fletcher32 are built in)"
                )))
            }
        };
    }
    Ok(buf)
}

fn unshuffle(buf: &[u8], esz: usize) -> Vec<u8> {
    if esz <= 1 {
        return buf.to_vec();
    }
    let n = buf.len() / esz;
    let mut out = vec![0u8; buf.len()];
    for i in 0..esz {
        for j in 0..n {
            out[j * esz + i] = buf[i * n + j];
        }
    }
    out
}

/// Read and de-filter the whole payload of a dataset.
pub fn raw_bytes(file: &[u8], sb: &Superblock, d: &DatasetInfo) -> Result<Vec<u8>> {
    let esz = d.dtype.size().ok_or_else(|| Error::unsupported("variable-size HDF5 datatype"))?;
    let total: usize = d.shape.iter().product::<usize>() * esz;
    match &d.layout {
        Layout::Compact(b) => Ok(b.clone()),
        Layout::Contiguous { address, size } => {
            if sb.undefined(*address) {
                return Ok(vec![0u8; total]);
            }
            // The layout message records the stored extent; if it disagrees
            // with what the dataspace implies, the file is inconsistent.
            if (*size as usize) < total {
                return Err(Error::format(format!(
                    "contiguous dataset stores {size} bytes but its shape needs {total}"
                )));
            }
            let p = sb.at(*address);
            Ok(file
                .get(p..p + total)
                .ok_or_else(|| Error::format("contiguous dataset runs past EOF"))?
                .to_vec())
        }
        Layout::Chunked { address, dims, elem_size } => {
            let mut out = vec![0u8; total];
            if sb.undefined(*address) {
                return Ok(out);
            }
            let mut chunks = Vec::new();
            btree_chunks(file, sb, *address, dims.len(), &mut chunks)?;
            for (offsets, addr, size, _mask) in chunks {
                let p = sb.at(addr);
                let raw = file
                    .get(p..p + size)
                    .ok_or_else(|| Error::format("chunk runs past EOF"))?
                    .to_vec();
                let chunk_elems: usize = dims.iter().product();
                let data = unfilter(raw, &d.filters, chunk_elems * *elem_size, *elem_size)?;
                scatter(&mut out, &data, &offsets, dims, &d.shape, esz);
            }
            Ok(out)
        }
    }
}
