//! Loading the `tenferro-einsum-benchmark` instance corpus and translating one
//! instance into cpueinsum's integer-label, SSA-order API.
//!
//! The corpus is one JSON file per problem. Its `format_string` and `shapes`
//! are the row-major description (the files also carry a column-major mirror,
//! `format_string_colmajor`/`shapes_colmajor`, of the same contraction), and
//! `paths.opt_size`/`paths.opt_flops` are opt_einsum-style *positional* paths:
//! each step names two positions in the shrinking operand list, with the
//! result appended at the end.
//!
//! cpueinsum wants the other form: integer labels, an output, and a path in
//! **SSA form** where step `k` contracts two existing operands into operand
//! `n + k`. [`to_ssa`] replays the positional list once to convert. The labels
//! each intermediate keeps are not a parameter of `EinsumSpec`; cpueinsum
//! derives them itself at plan time from the description and the live-label
//! count (`crates/cpueinsum/src/plan.rs:325`), so the translation supplies the
//! inputs, the output and the order only.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use cpueinsum::EinsumSpec;

use crate::json::{self, Json};

/// The two contraction orders the corpus declares, one per engine row.
pub const ENGINE_ROWS: [&str; 2] = ["opt_size", "opt_flops"];

#[cfg(test)]
mod tests;

/// One corpus problem, loaded and (best effort) translated.
pub struct Instance {
    /// The instance file's `name`, which is the harness's `--family` id.
    pub name: String,
    /// The dtype label as the corpus declares it (`float64`).
    pub dtype: String,
    /// Operand count as the corpus declares it.
    pub operands: usize,
    /// Total elements over all input operands.
    pub input_len: usize,
    /// Output element count.
    pub output_len: usize,
    /// The translated problem, or why the description could not be translated.
    pub translation: Result<Translation, String>,
}

/// A translated instance: integer labels, layouts and the declared orders.
#[derive(Debug)]
pub struct Translation {
    /// Integer labels of each input.
    pub inputs: Vec<Vec<i64>>,
    /// Integer labels of the output.
    pub output: Vec<i64>,
    /// Row-major extents of each input.
    pub dims: Vec<Vec<usize>>,
    /// Row-major strides of each input.
    pub strides: Vec<Vec<isize>>,
    /// Row-major extents of the output.
    pub out_dims: Vec<usize>,
    /// Row-major strides of the output.
    pub out_strides: Vec<isize>,
    /// Output element count.
    pub out_len: usize,
    /// The declared orders, still positional, keyed by engine row.
    raw_paths: BTreeMap<String, Vec<[usize; 2]>>,
}

impl Instance {
    /// Every `*.json` file in `dir`, sorted by name so the grid is stable.
    pub fn load_dir(dir: &Path) -> Result<Vec<Instance>, String> {
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .map(|entry| entry.map(|e| e.path()).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        files.retain(|path| path.extension().and_then(|s| s.to_str()) == Some("json"));
        files.sort();
        if files.is_empty() {
            return Err(format!("no *.json instance files in {}", dir.display()));
        }
        files.iter().map(|path| Instance::load(path)).collect()
    }

    fn load(path: &Path) -> Result<Instance, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let doc = json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let field = |key: &str| -> Result<&Json, String> {
            doc.get(key)
                .ok_or_else(|| format!("{}: missing {key}", path.display()))
        };
        let name = field("name")?
            .as_str()
            .ok_or("name is not a string")?
            .to_string();
        let dtype = field("dtype")?
            .as_str()
            .ok_or("dtype is not a string")?
            .to_string();
        let format = field("format_string")?
            .as_str()
            .ok_or("format_string is not a string")?;
        let shapes = parse_shapes(field("shapes")?)?;
        let raw_paths = parse_paths(field("paths")?)?;

        let operands = shapes.len();
        let input_len = shapes.iter().map(|d| d.iter().product::<usize>()).sum();
        let translation = Translation::new(format, &shapes, raw_paths);
        let output_len = match &translation {
            Ok(t) => t.out_len,
            Err(_) => 0,
        };
        Ok(Instance {
            name,
            dtype,
            operands,
            input_len,
            output_len,
            translation,
        })
    }

    /// The `EinsumSpec` for one engine row, or why this row cannot be built.
    pub fn spec(&self, row: &str) -> Result<EinsumSpec, String> {
        let translation = self.translation.as_ref().map_err(|e| e.clone())?;
        translation.spec(row)
    }
}

impl Translation {
    fn new(
        format: &str,
        shapes: &[Vec<usize>],
        raw_paths: BTreeMap<String, Vec<[usize; 2]>>,
    ) -> Result<Translation, String> {
        let (lhs, rhs) = format
            .split_once("->")
            .ok_or_else(|| format!("format string {format:?} has no '->'"))?;
        let input_strs: Vec<&str> = lhs.split(',').collect();
        if input_strs.is_empty() || input_strs.len() != shapes.len() {
            return Err(format!(
                "format string has {} inputs but the file declares {} shapes",
                input_strs.len(),
                shapes.len()
            ));
        }

        // Intern each distinct label char to one integer id, so equal chars are
        // equal labels.
        let mut ids: Vec<char> = Vec::new();
        let mut intern = |c: char| -> i64 {
            match ids.iter().position(|&x| x == c) {
                Some(k) => k as i64,
                None => {
                    ids.push(c);
                    (ids.len() - 1) as i64
                }
            }
        };
        let inputs: Vec<Vec<i64>> = input_strs
            .iter()
            .map(|s| s.chars().map(&mut intern).collect())
            .collect();
        let output: Vec<i64> = rhs.chars().map(&mut intern).collect();

        // One extent per label; a conflict is a malformed instance.
        let mut extent: Vec<usize> = vec![usize::MAX; ids.len()];
        for (labels, shape) in inputs.iter().zip(shapes) {
            if labels.len() != shape.len() {
                return Err(format!(
                    "operand rank {} does not match its {} shape axes",
                    labels.len(),
                    shape.len()
                ));
            }
            for (&label, &axis) in labels.iter().zip(shape) {
                let slot = &mut extent[label as usize];
                if *slot == usize::MAX {
                    *slot = axis;
                } else if *slot != axis {
                    return Err(format!(
                        "label {:?} has extent {axis} here and {slot} elsewhere",
                        ids[label as usize]
                    ));
                }
            }
        }
        if output.iter().any(|&l| extent[l as usize] == usize::MAX) {
            return Err("the output names a label no input carries".to_string());
        }

        let dims: Vec<Vec<usize>> = shapes.to_vec();
        let strides: Vec<Vec<isize>> = dims.iter().map(|d| row_major(d).0).collect();
        let out_dims: Vec<usize> = output.iter().map(|&l| extent[l as usize]).collect();
        let out_strides = row_major(&out_dims).0;
        let out_len = out_dims.iter().product();

        Ok(Translation {
            inputs,
            output,
            dims,
            strides,
            out_dims,
            out_strides,
            out_len,
            raw_paths,
        })
    }

    fn spec(&self, row: &str) -> Result<EinsumSpec, String> {
        let path = self
            .raw_paths
            .get(row)
            .ok_or_else(|| format!("the corpus declares no path {row:?}"))?;
        let ssa = to_ssa(path, self.inputs.len())?;
        let inputs: Vec<&[i64]> = self.inputs.iter().map(Vec::as_slice).collect();
        EinsumSpec::new(&inputs, &self.output, &ssa).map_err(|e| e.to_string())
    }
}

/// Row-major strides and the element count of `dims`.
fn row_major(dims: &[usize]) -> (Vec<isize>, usize) {
    let mut strides = vec![0isize; dims.len()];
    let mut len = 1usize;
    for axis in (0..dims.len()).rev() {
        strides[axis] = len as isize;
        len *= dims[axis];
    }
    (strides, len)
}

/// Replay an opt_einsum positional path into SSA form.
///
/// The list starts as the `n` inputs. Step `k` pairs positions `a` and `b` in
/// the current list; the SSA step is the pair of operand ids those positions
/// hold, and the result becomes operand `n + k`, appended at the end after the
/// two consumed entries are removed.
fn to_ssa(path: &[[usize; 2]], n: usize) -> Result<Vec<[usize; 2]>, String> {
    if path.len() != n.saturating_sub(1) {
        return Err(format!(
            "the declared path has {} steps but {n} inputs need {}",
            path.len(),
            n.saturating_sub(1)
        ));
    }
    let mut list: Vec<usize> = (0..n).collect();
    let mut ssa = Vec::with_capacity(path.len());
    for (step, &[a, b]) in path.iter().enumerate() {
        if a >= list.len() || b >= list.len() {
            return Err(format!(
                "path step {step} names positions {a},{b} of {}",
                list.len()
            ));
        }
        if a == b {
            return Err(format!("path step {step} names position {a} twice"));
        }
        ssa.push([list[a], list[b]]);
        let (hi, lo) = if a > b { (a, b) } else { (b, a) };
        list.remove(hi);
        list.remove(lo);
        list.push(n + step);
    }
    Ok(ssa)
}

/// The `shapes` array: an array of arrays of non-negative integers.
fn parse_shapes(value: &Json) -> Result<Vec<Vec<usize>>, String> {
    let rows = value.as_arr().ok_or("shapes is not an array")?;
    rows.iter()
        .map(|row| {
            row.as_arr()
                .ok_or("a shapes entry is not an array")?
                .iter()
                .map(|axis| {
                    let n = axis.as_f64().ok_or("a shape axis is not a number")?;
                    if n < 0.0 || n.fract() != 0.0 {
                        return Err(format!("shape axis {n} is not a non-negative integer"));
                    }
                    Ok(n as usize)
                })
                .collect()
        })
        .collect()
}

/// The `paths` object: `{ row: { path: [[a, b], ...], ... }, ... }`.
fn parse_paths(value: &Json) -> Result<BTreeMap<String, Vec<[usize; 2]>>, String> {
    let object = value.as_obj().ok_or("paths is not an object")?;
    let mut paths = BTreeMap::new();
    for (row, entry) in object {
        let path = entry
            .get("path")
            .ok_or_else(|| format!("paths.{row} has no path"))?
            .as_arr()
            .ok_or_else(|| format!("paths.{row}.path is not an array"))?;
        let mut steps = Vec::with_capacity(path.len());
        for step in path {
            let pair = step
                .as_arr()
                .ok_or_else(|| format!("paths.{row}.path has a non-array step"))?;
            if pair.len() != 2 {
                return Err(format!(
                    "paths.{row}.path has a step with {} entries",
                    pair.len()
                ));
            }
            let index = |v: &Json| -> Result<usize, String> {
                let n = v.as_f64().ok_or("a path index is not a number")?;
                if n < 0.0 || n.fract() != 0.0 {
                    return Err(format!("path index {n} is not a non-negative integer"));
                }
                Ok(n as usize)
            };
            steps.push([index(&pair[0])?, index(&pair[1])?]);
        }
        paths.insert(row.clone(), steps);
    }
    Ok(paths)
}
