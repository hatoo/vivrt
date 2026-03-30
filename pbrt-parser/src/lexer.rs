use crate::error::{ParseError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    String(String),
    Number(f64),
    Bool(bool),
    LBracket,
    RBracket,
    Identifier(String),
}

#[derive(Debug, Clone)]
pub struct Located<T> {
    pub value: T,
    pub line: usize,
    pub column: usize,
}

pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Located<Token>>> {
        let mut tokens = Vec::new();
        while let Some(tok) = self.next_token()? {
            tokens.push(tok);
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.input.get(self.pos).copied()?;
        self.pos += 1;
        if ch == b'\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => {
                    self.advance();
                }
                Some(b'#') => {
                    // Skip to end of line
                    while let Some(ch) = self.advance() {
                        if ch == b'\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn next_token(&mut self) -> Result<Option<Located<Token>>> {
        self.skip_whitespace_and_comments();

        let Some(ch) = self.peek() else {
            return Ok(None);
        };

        let line = self.line;
        let column = self.column;

        let token = match ch {
            b'[' => {
                self.advance();
                Token::LBracket
            }
            b']' => {
                self.advance();
                Token::RBracket
            }
            b'"' => Token::String(self.read_string()?),
            b'-' | b'+' | b'.' | b'0'..=b'9' => Token::Number(self.read_number()?),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "true" => Token::Bool(true),
                    "false" => Token::Bool(false),
                    _ => Token::Identifier(ident),
                }
            }
            _ => {
                return Err(ParseError::new(
                    format!("unexpected character '{}'", ch as char),
                    line,
                    column,
                ));
            }
        };

        Ok(Some(Located {
            value: token,
            line,
            column,
        }))
    }

    fn read_string(&mut self) -> Result<String> {
        let line = self.line;
        let column = self.column;
        self.advance(); // skip opening quote
        let mut s = Vec::new();
        loop {
            match self.advance() {
                Some(b'"') => break,
                Some(b'\\') => match self.advance() {
                    Some(b'n') => s.push(b'\n'),
                    Some(b't') => s.push(b'\t'),
                    Some(b'\\') => s.push(b'\\'),
                    Some(b'"') => s.push(b'"'),
                    Some(ch) => {
                        s.push(b'\\');
                        s.push(ch);
                    }
                    None => {
                        return Err(ParseError::new("unterminated string", line, column));
                    }
                },
                Some(ch) => s.push(ch),
                None => {
                    return Err(ParseError::new("unterminated string", line, column));
                }
            }
        }
        String::from_utf8(s).map_err(|_| ParseError::new("invalid UTF-8 in string", line, column))
    }

    fn read_number(&mut self) -> Result<f64> {
        let line = self.line;
        let column = self.column;
        let start = self.pos;

        // Sign
        if matches!(self.peek(), Some(b'-' | b'+')) {
            self.advance();
        }

        // Integer part
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.advance();
        }

        // Fractional part
        if matches!(self.peek(), Some(b'.')) {
            self.advance();
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.advance();
            }
        }

        // Exponent
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.advance();
            if matches!(self.peek(), Some(b'-' | b'+')) {
                self.advance();
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.advance();
            }
        }

        let s = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
        s.parse::<f64>()
            .map_err(|_| ParseError::new(format!("invalid number '{s}'"), line, column))
    }

    fn read_identifier(&mut self) -> String {
        let start = self.pos;
        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
        ) {
            self.advance();
        }
        String::from_utf8_lossy(&self.input[start..self.pos]).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(input: &str) -> Vec<Token> {
        Lexer::new(input)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.value)
            .collect()
    }

    #[test]
    fn test_basic_tokens() {
        assert_eq!(
            tokens(r#"Shape "sphere" "float radius" [1.5]"#),
            vec![
                Token::Identifier("Shape".into()),
                Token::String("sphere".into()),
                Token::String("float radius".into()),
                Token::LBracket,
                Token::Number(1.5),
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn test_comments() {
        assert_eq!(
            tokens("Identity # reset\nWorldBegin"),
            vec![
                Token::Identifier("Identity".into()),
                Token::Identifier("WorldBegin".into()),
            ]
        );
    }

    #[test]
    fn test_booleans() {
        assert_eq!(
            tokens(r#""bool renderquickly" true"#),
            vec![
                Token::String("bool renderquickly".into()),
                Token::Bool(true),
            ]
        );
    }

    #[test]
    fn test_negative_number() {
        assert_eq!(tokens("-1.5"), vec![Token::Number(-1.5)]);
    }

    #[test]
    fn test_scientific_notation() {
        assert_eq!(tokens("1e16"), vec![Token::Number(1e16)]);
        assert_eq!(tokens("1.5E-3"), vec![Token::Number(1.5e-3)]);
    }
}
