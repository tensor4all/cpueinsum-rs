//! A minimal JSON reader for the instance corpus.
//!
//! The workspace carries no JSON crate and this harness may not add one, while
//! the corpus has a fixed, trusted schema: one object per file holding the
//! format string, the shapes and the two declared contraction paths. A small
//! recursive descent reader is enough.

use std::collections::BTreeMap;

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A number, kept as `f64`; the corpus extents are small integers.
    Num(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object.
    Obj(BTreeMap<String, Json>),
}

impl Json {
    /// The value of an object member, or `None` for a non-object or a missing key.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(map) => map.get(key),
            _ => None,
        }
    }

    /// The string, or `None` for another type.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The number, or `None` for another type.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// The array elements, or `None` for another type.
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    /// The object members, or `None` for another type.
    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(map) => Some(map),
            _ => None,
        }
    }
}

/// Parse `text`, rejecting trailing content.
pub fn parse(text: &str) -> Result<Json, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    parser.ws();
    let value = parser.value()?;
    parser.ws();
    if parser.at != parser.bytes.len() {
        return Err(format!("trailing content at byte {}", parser.at));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", byte as char, self.at))
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("expected {word:?} at byte {}", self.at))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek().ok_or("unexpected end of input")? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            b'n' => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Obj(map));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            map.insert(key, self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.at)),
            }
        }
        Ok(Json::Obj(map))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.at)),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let byte = *self.bytes.get(self.at).ok_or("unterminated string")?;
            self.at += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escape = *self.bytes.get(self.at).ok_or("unterminated escape")?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0c),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let ch = self.unicode()?;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => {
                            return Err(format!(
                                "invalid escape \\{} at byte {}",
                                escape as char,
                                self.at - 2
                            ))
                        }
                    }
                }
                _ => out.push(byte),
            }
        }
        String::from_utf8(out).map_err(|e| format!("string is not UTF-8: {e}"))
    }

    fn unicode(&mut self) -> Result<char, String> {
        let high = self.hex4()?;
        let point = if (0xD800..0xDC00).contains(&high) {
            // A surrogate pair is written as two adjacent `\u` escapes.
            if self.bytes.get(self.at) == Some(&b'\\') && self.bytes.get(self.at + 1) == Some(&b'u')
            {
                self.at += 2;
                let low = self.hex4()?;
                if !(0xDC00..0xE000).contains(&low) {
                    return Err(format!("bad low surrogate {low:04x}"));
                }
                0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
            } else {
                return Err("high surrogate without a pair".into());
            }
        } else if (0xDC00..0xE000).contains(&high) {
            return Err("lone low surrogate".into());
        } else {
            high
        };
        char::from_u32(point).ok_or_else(|| format!("invalid code point {point:04x}"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = *self.bytes.get(self.at).ok_or("truncated \\u escape")?;
            self.at += 1;
            let digit = (byte as char)
                .to_digit(16)
                .ok_or_else(|| format!("bad hex digit {:?}", byte as char))?;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.at += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|e| e.to_string())?;
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("invalid number {text:?} at byte {start}"))
    }
}
