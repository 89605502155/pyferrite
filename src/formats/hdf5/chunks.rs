//! Version-1 chunk B-trees and the scatter of chunks into a dense buffer.

use super::superblock::Superblock;
use crate::error::{Error, Result};
use crate::util::Cursor;

/// Copy one chunk into the right place of the dense output buffer.
pub(super) fn scatter(
    out: &mut [u8],
    chunk: &[u8],
    offsets: &[usize],
    dims: &[usize],
    shape: &[usize],
    esz: usize,
) {
    let rank = shape.len();
    if rank == 0 {
        let n = esz.min(chunk.len()).min(out.len());
        out[..n].copy_from_slice(&chunk[..n]);
        return;
    }
    let row = dims[rank - 1];
    let rows: usize = dims[..rank - 1].iter().product();
    let mut idx = vec![0usize; rank.saturating_sub(1)];
    for r in 0..rows {
        let mut flat = 0usize;
        let mut stride = 1usize;
        // C-order flat index of (offsets + idx, 0)
        for d in (0..rank).rev() {
            let coord = if d == rank - 1 { offsets[d] } else { offsets[d] + idx[d] };
            if coord >= shape[d] {
                flat = usize::MAX;
                break;
            }
            flat = flat.saturating_add(coord * stride);
            stride *= shape[d];
        }
        if flat != usize::MAX {
            let avail = shape[rank - 1].saturating_sub(offsets[rank - 1]).min(row);
            let src = r * row * esz;
            let dst = flat * esz;
            if src + avail * esz <= chunk.len() && dst + avail * esz <= out.len() {
                out[dst..dst + avail * esz].copy_from_slice(&chunk[src..src + avail * esz]);
            }
        }
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

type ChunkRec = (Vec<usize>, u64, usize, u32);

/// Walk a version-1 chunk B-tree, collecting every leaf chunk.
pub(super) fn btree_chunks(
    file: &[u8],
    sb: &Superblock,
    addr: u64,
    rank: usize,
    out: &mut Vec<ChunkRec>,
) -> Result<()> {
    let p = sb.at(addr);
    let mut c = Cursor::at(file, p);
    if c.take(4)? != b"TREE" {
        return Err(Error::format("bad chunk B-tree signature"));
    }
    let node_type = c.u8()?;
    let level = c.u8()?;
    let entries = c.u16()? as usize;
    c.skip(2 * sb.offset_size)?;
    if node_type != 1 {
        return Err(Error::format("expected a chunk B-tree"));
    }
    let mut keys = Vec::with_capacity(entries);
    let mut kids = Vec::with_capacity(entries);
    for _ in 0..entries {
        let size = c.u32()? as usize;
        let mask = c.u32()?;
        let mut offs = Vec::with_capacity(rank + 1);
        for _ in 0..=rank {
            offs.push(c.u64()? as usize);
        }
        offs.pop();
        keys.push((offs, size, mask));
        kids.push(c.uint(sb.offset_size)?);
    }
    for (i, k) in kids.into_iter().enumerate() {
        if level > 0 {
            btree_chunks(file, sb, k, rank, out)?;
        } else {
            let (offs, size, mask) = keys[i].clone();
            out.push((offs, k, size, mask));
        }
    }
    Ok(())
}
