//! HDF5 object headers, versions 1 and 2 (including continuation blocks).

use super::superblock::Superblock;
use crate::error::{Error, Result};
use crate::util::Cursor;

/// One header message: its type tag and its raw payload.
#[derive(Clone, Debug)]
pub struct Message {
    pub kind: u16,
    pub data: Vec<u8>,
}

/// Parse the object header at `addr`, following continuation blocks.
pub fn parse(file: &[u8], sb: &Superblock, addr: u64) -> Result<Vec<Message>> {
    let mut out = Vec::new();
    let mut queue = vec![(sb.at(addr), None::<usize>)];
    let mut guard = 0;
    while let Some((pos, limit)) = queue.pop() {
        guard += 1;
        if guard > 256 {
            return Err(Error::format("object header continuation loop"));
        }
        if file.get(pos..pos + 4).map(|s| s == b"OHDR").unwrap_or(false) {
            parse_v2(file, sb, pos, &mut out, &mut queue)?;
        } else if file.get(pos..pos + 4).map(|s| s == b"OCHK").unwrap_or(false) {
            parse_v2_chunk(file, sb, pos + 4, limit, &mut out, &mut queue)?;
        } else if let Some(limit) = limit {
            parse_v1_messages(file, sb, pos, limit, &mut out, &mut queue)?;
        } else {
            parse_v1(file, sb, pos, &mut out, &mut queue)?;
        }
    }
    Ok(out)
}

fn parse_v1(
    file: &[u8],
    sb: &Superblock,
    pos: usize,
    out: &mut Vec<Message>,
    queue: &mut Vec<(usize, Option<usize>)>,
) -> Result<()> {
    let mut c = Cursor::at(file, pos);
    let version = c.u8()?;
    if version != 1 {
        return Err(Error::unsupported(format!("object header version {version}")));
    }
    c.skip(1)?;
    let _nmsgs = c.u16()?;
    let _refcount = c.u32()?;
    let size = c.u32()? as usize;
    // The prefix is twelve bytes plus four of padding, so messages always begin
    // sixteen bytes in. Aligning the *absolute* cursor would be wrong: the
    // library is free to place a header at an unaligned address, and does.
    let start = pos + 16;
    parse_v1_messages(file, sb, start, start + size, out, queue)
}

fn parse_v1_messages(
    file: &[u8],
    sb: &Superblock,
    start: usize,
    end: usize,
    out: &mut Vec<Message>,
    queue: &mut Vec<(usize, Option<usize>)>,
) -> Result<()> {
    let mut c = Cursor::at(file, start);
    while c.pos() + 8 <= end.min(file.len()) {
        let kind = c.u16()?;
        let size = c.u16()? as usize;
        let _flags = c.u8()?;
        c.skip(3)?;
        if c.pos() + size > file.len() {
            break;
        }
        let data = c.take(size)?.to_vec();
        if kind == super::msg::MSG_CONTINUATION {
            let mut cc = Cursor::new(&data);
            let a = cc.uint(sb.offset_size)?;
            let l = cc.uint(sb.length_size)? as usize;
            if sb.in_bounds(a) {
                let p = sb.at(a);
                queue.push((p, Some(p + l)));
            }
        } else if kind != 0 {
            out.push(Message { kind, data });
        }
    }
    Ok(())
}

fn parse_v2(
    file: &[u8],
    sb: &Superblock,
    pos: usize,
    out: &mut Vec<Message>,
    queue: &mut Vec<(usize, Option<usize>)>,
) -> Result<()> {
    let mut c = Cursor::at(file, pos + 4);
    let version = c.u8()?;
    if version != 2 {
        return Err(Error::unsupported(format!("OHDR version {version}")));
    }
    let flags = c.u8()?;
    if flags & 0x20 != 0 {
        c.skip(16)?; // access/mod/change/birth times
    }
    if flags & 0x10 != 0 {
        c.skip(4)?; // max compact / min dense
    }
    let size = match flags & 0x03 {
        0 => c.u8()? as usize,
        1 => c.u16()? as usize,
        2 => c.u32()? as usize,
        _ => c.u64()? as usize,
    };
    let start = c.pos();
    parse_v2_body(file, sb, start, start + size, flags, out, queue)
}

fn parse_v2_chunk(
    file: &[u8],
    sb: &Superblock,
    start: usize,
    limit: Option<usize>,
    out: &mut Vec<Message>,
    queue: &mut Vec<(usize, Option<usize>)>,
) -> Result<()> {
    let end = limit.unwrap_or(file.len()).min(file.len());
    parse_v2_body(file, sb, start, end.saturating_sub(4), 0, out, queue)
}

fn parse_v2_body(
    file: &[u8],
    sb: &Superblock,
    start: usize,
    end: usize,
    flags: u8,
    out: &mut Vec<Message>,
    queue: &mut Vec<(usize, Option<usize>)>,
) -> Result<()> {
    let mut c = Cursor::at(file, start);
    let track_order = flags & 0x04 != 0;
    while c.pos() + 4 <= end.min(file.len()) {
        let kind = c.u8()? as u16;
        let size = c.u16()? as usize;
        let _mflags = c.u8()?;
        if track_order {
            c.skip(2)?;
        }
        if c.pos() + size > file.len() {
            break;
        }
        let data = c.take(size)?.to_vec();
        if kind == super::msg::MSG_CONTINUATION {
            let mut cc = Cursor::new(&data);
            let a = cc.uint(sb.offset_size)?;
            let l = cc.uint(sb.length_size)? as usize;
            if sb.in_bounds(a) {
                let p = sb.at(a);
                queue.push((p, Some(p + l)));
            }
        } else if kind != 0 {
            out.push(Message { kind, data });
        }
    }
    Ok(())
}
